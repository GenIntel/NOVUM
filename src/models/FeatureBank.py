import math

import torch
import torch.nn.functional as F
from torch import nn


def one_hot(y, max_size=None):
    if not max_size:
        max_size = int(torch.max(y).item() + 1)
    y = y.view(-1, 1)
    y_onehot = torch.zeros((y.shape[0], max_size), dtype=torch.float32, device=y.device)
    y_onehot.scatter_(1, y.type(torch.long), 1)
    return y_onehot


def mask_remove_near(
    keypoints,
    thr,
    img_label,
    pad_index,
    zeros,
    nb_classes,
    dtype_template=None,
    num_neg=0,
    neg_weight=1,
    eps=1e5,
):
    if dtype_template is None:
        dtype_template = torch.ones(1, dtype=torch.float32)

    if dtype_template.dtype != zeros.dtype:
        zeros = zeros.type_as(dtype_template)

    # keypoints -> [n, k, 2]
    with torch.no_grad():
        # distance -> [n, k, k]
        distance = torch.sum(
            (torch.unsqueeze(keypoints, dim=1) - torch.unsqueeze(keypoints, dim=2)).pow(
                2,
            ),
            dim=3,
        ).pow(0.5)
        if num_neg == 0:
            return (
                (distance <= thr).type_as(dtype_template)
                - torch.eye(keypoints.shape[1]).type_as(dtype_template).unsqueeze(dim=0)
            ) * eps
        else:
            tem = (distance <= thr).type_as(dtype_template) - torch.eye(
                keypoints.shape[1],
            ).type_as(dtype_template).unsqueeze(dim=0)
            # TODO: Change its dimension according to current bank, and also sets values of padded keypoints in 2nd dimension to large nums.

            if zeros.shape[0] != keypoints.shape[0]:
                zeros = torch.zeros(
                    tem.shape[0],
                    tem.shape[1],
                    tem.shape[1] * nb_classes,
                ).type_as(dtype_template)

            for i in range(tem.shape[0]):
                zeros[
                    i,
                    :,
                    img_label[i] * tem.shape[1] : (img_label[i] + 1) * tem.shape[1],
                ] = (
                    tem[i] * eps
                )
            # for i in range(12):
            # ret[:, :, tem.shape[1] * i + n_list[i] : tem.shape[1] * (i+1)] = eps
            zeros[:, :, pad_index.view(-1)] = eps
            # what is concated is the clutter part, does not need to change
            return torch.cat(
                [
                    zeros,
                    -torch.ones(keypoints.shape[0:2] + (num_neg,)).type_as(
                        dtype_template,
                    )
                    * math.log(neg_weight),
                ],
                dim=2,
            )


def fun_label_onehot(img_label, count_label, nb_classes):
    ret = torch.zeros(img_label.shape[0], nb_classes).to(img_label.device)
    ret = ret.scatter_(1, img_label.unsqueeze(1), 1.0).to(img_label.device)
    for i in range(nb_classes):
        count = count_label[i]
        if count == 0:
            continue
        ret[:, i] /= count
    return ret


class FeatureBank(nn.Module):
    def __init__(
        self,
        inputSize,
        outputSize,
        num_pos,
        T=0.07,
        momentum=0.5,
        max_groups=-1,
        num_noise=-1,
        nb_classes=12,
    ):
        super().__init__()
        self.nLem = outputSize

        self.register_buffer("params", torch.tensor([1, T, -1, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.nb_classes = nb_classes
        self.single_feature_dim = int(num_pos / self.nb_classes)

        self.memory = torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        self.memory.requires_grad = False

        self.lru = 0
        if max_groups > 0:
            self.max_lru = max_groups
        else:
            self.max_lru = -1

        if num_noise < 0:
            self.num_noise = 1
        else:
            self.num_noise = num_noise

        self.num_pos = num_pos

    @property
    def features(self):
        return self.memory[:self.num_pos]
    
    @property
    def clutter(self):
        return self.memory[self.num_pos:]

    def features_of(self, class_, n_list_set):
        start = class_*self.single_feature_dim
        return self.memory[start:start+n_list_set[class_]]

    def class_id_from_vertex_id(self, vertex_id):
        return vertex_id // self.single_feature_dim

    def cuboid_vertex_id_from_vertex_id(self, vertex_id):
        return vertex_id % self.single_feature_dim

    # x: feature: [128, 128], y: indexes [128] -- a batch of data's index directly from the dataloader.
    def forward(self, x, y, visible, img_label):
        # k = feature_space
        # n = batch_size
        n_pos = self.num_pos
        n_neg = self.num_noise
        count_label = torch.bincount(img_label, minlength=self.nb_classes)
        label_weight_onehot = fun_label_onehot(img_label, count_label, self.nb_classes)

        # set max_group is it is explicitly given. (We give max_group as 512, so not in our case)
        if (
            self.max_lru == -1
            and n_neg > 0
            and x.shape[0] <= (self.nLem - n_pos) / n_neg
        ):
            self.max_lru = (self.memory.shape[0] - n_pos) // (n_neg * x.shape[0])

        momentum = self.params[3].item()

        # x [n, k, d] * memory [l, d] = similarity : [n, k, l]
        if n_neg == 0:
            similarity_to_full_memory = torch.matmul(
                x,
                torch.transpose(self.memory, 0, 1),
            )
            noise_similarity_to_features = torch.zeros(1)
        else:
            # first dimension is batch, get first n space that represnets keypoints
            t_ = x[:, 0 : self.single_feature_dim, :]
            # calculate similarity that includes clutter in memory bank
            similarity_to_full_memory = torch.matmul(
                t_,
                torch.transpose(self.memory, 0, 1),
            )
            # calculate the space after first n space that serves as noise for noise similarity calculation, calculated with feature part of memory
            noise_similarity_to_features = torch.matmul(
                x[:, self.single_feature_dim :, :],
                torch.transpose(self.memory[0:n_pos, :], 0, 1),
            )

        with torch.set_grad_enabled(False):
            # [n, k, k]
            # # change label to one-hot format. 1 at diagonal.
            # y_onehot = one_hot(y, n_class).view(x.shape[0], -1, n_class)
            y_idx = y.type(torch.long)

            get = torch.matmul(
                label_weight_onehot.transpose(0, 1),
                (
                    x[:, 0 : self.single_feature_dim, :]
                    * visible.type(x.dtype).view(*visible.shape, 1)
                ).view(x.shape[0], -1),
            )
            get = get.view(get.shape[0], -1, x.shape[-1])
            # handle 0 in get, case that no img of one class is in the batch
            tmp = (count_label == 0).nonzero(as_tuple=True)[0]
            for i in tmp:
                # copy memory to get
                get[i] = self.memory[
                    i * self.single_feature_dim : (i + 1) * self.single_feature_dim
                ]
            get = get.view(-1, x.shape[-1])

            if n_neg > 0:
                if x.shape[0] > (self.nLem - n_pos) / n_neg:
                    self.memory = F.normalize(
                        torch.cat(
                            [
                                self.memory[0:n_pos, :] * momentum
                                + get * (1 - momentum),
                                x[:, self.single_feature_dim : :, :]
                                .contiguous()
                                .view(-1, x.shape[2])[0 : self.memory.shape[0] - n_pos],
                            ],
                            dim=0,
                        ),
                        dim=1,
                        p=2,
                    )
                else:
                    # handle case if batchsize is not larger than max_group
                    # neg_parts updated based on tagging
                    neg_parts = torch.cat(
                        [
                            self.memory[
                                n_pos : n_pos + self.lru * n_neg * x.shape[0],
                                :,
                            ],
                            x[:, self.single_feature_dim : :, :]
                            .contiguous()
                            .view(-1, x.shape[2]),
                            self.memory[
                                n_pos + (self.lru + 1) * n_neg * x.shape[0] : :,
                                :,
                            ],
                        ],
                        dim=0,
                    )

                    self.memory = F.normalize(
                        torch.cat(
                            [
                                self.memory[0:n_pos, :] * momentum
                                + get * (1 - momentum),
                                neg_parts,
                            ],
                            dim=0,
                        ),
                        dim=1,
                        p=2,
                    )
            else:
                self.memory = F.normalize(
                    self.memory[0:n_pos, :] * momentum + get * (1 - momentum),
                    dim=1,
                    p=2,
                )

            self.lru += 1
            self.lru = self.lru % self.max_lru
        # out.shape: [d, n_neg + n_pos]
        return (
            similarity_to_full_memory,
            y_idx,
            noise_similarity_to_features,
            label_weight_onehot,
        )

    def set_zero(self, n_pos):
        self.accumulate_num = torch.zeros(
            n_pos,
            dtype=torch.long,
            device=self.memory.device,
        )
        self.memory.fill_(0)

    def accumulate_memory(self, x, y, visible, eps=1e-8):
        n_pos = self.num_pos
        n_neg = self.num_noise

        with torch.no_grad():
            # update memory keypoints
            # [n, k, k]
            idx_onehot = one_hot(y, n_pos).view(x.shape[0], -1, n_pos)

            # [k, d]
            get = torch.bmm(
                torch.transpose(idx_onehot, 1, 2),
                x[:, 0:n_pos, :] * visible.type(x.dtype).view(*visible.shape, 1),
            )
            get = torch.sum(get, dim=0)

            self.memory[0:n_pos, :].copy_(self.memory[0:n_pos, :] + get)
            self.accumulate_num += torch.sum(
                visible.type(self.accumulate_num.dtype),
                dim=0,
            )

    def normalize_memory(self):
        self.memory.copy_(F.normalize(self.memory, p=2, dim=1))

    def cuda(self, device=None):
        super().cuda(device)
        self.memory = self.memory.cuda(device)
        return self
    
    def load_memory(self, memory):
        self.memory = memory
