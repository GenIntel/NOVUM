import os
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset.Pascal3DPlus import Normalize
from dataset.Pascal3DPlus import Pascal3DPlus
from dataset.Pascal3DPlus import ToTensor
from lib.get_n_list import get_n_list
from models.FeatureBank import mask_remove_near
from models.FeatureBank import FeatureBank
from models.KeypointRepresentationNet import NetE2E
from tqdm import trange
from lib.config import load_config, parse_args

args = parse_args()
config = load_config(args, load_default_config=False, log_info=False)
n_gpus = torch.cuda.device_count()
local_size = [config.model.local_size, config.model.local_size]

bank_set = []
dataloader_set = []
n_list_set = []
mesh_path_set = []

if "%s" in config.dataset.paths.mesh:
    for class_ in config.dataset.classes:
        mesh_path = Path(config.dataset.root, config.dataset.paths.mesh % class_)
        mesh_path_set.append(mesh_path)
        n_list = get_n_list(mesh_path)
        n_list_set.append(n_list[0])


os.makedirs(config.save_dir, exist_ok=True)

net = NetE2E(
    net_type=config.training.backbone,
    local_size=local_size,
    output_dimension=config.training.d_feature,
    reduce_function=None,
    n_noise_points=config.training.num_noise,
    pretrain=True,
    noise_on_mask=False,
)
net.train()
if config.model.separate_bank:
    net = torch.nn.DataParallel(net.cuda(), device_ids=[i for i in range(n_gpus - 1)])
else:
    net = torch.nn.DataParallel(net.cuda())


transforms = transforms.Compose(
    [
        ToTensor(),
        Normalize(),
    ],
)

mesh_path = mesh_path_set[0]
max_n = max(n_list_set)
fbank = FeatureBank(
    inputSize=config.training.d_feature,
    outputSize=len(config.dataset.classes) * max_n + config.model.num_noise * config.model.max_group,
    num_noise=config.model.num_noise,
    num_pos=len(config.dataset.classes) * max_n,
    momentum=config.model.adj_momentum,
)
fbank = fbank.cuda()

dataset = Pascal3DPlus(
    transforms=transforms,
    rootpath=config.paths.root,
    mesh_path=mesh_path,
    anno_path=config.paths.annot,
    list_path=config.paths.img_list,
    weighted=True,
    max_n=max_n,
)

shared_dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size,
    shuffle=True,
    num_workers=config.training.workers,
)

criterion = torch.nn.CrossEntropyLoss(reduction="none").cuda()

iter_num = 0
optim = torch.optim.Adam(net.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
last_device = "cuda:%d" % (n_gpus - 1)
fbank = fbank.cuda(last_device)

pad_index = []
for i in range(len(config.dataset.classes)):
    num = (max_n * (i + 1)) - (max_n * i + n_list_set[i])
    for j in range(num):
        n = max_n * i + n_list_set[i] + j
        pad_index.append(n)

pad_index = torch.tensor(pad_index, dtype=torch.long)
zeros = torch.zeros(
    config.training.batch_size,
    max_n,
    max_n * len(config.dataset.classes),
    dtype=torch.float32,
).to(last_device)


def save_checkpoint(state, filename):
    file = os.path.join(config.training.save_dir, filename)
    torch.save(state, file)


print("Start Training!")
for epoch in trange(config.training.total_epochs):
    if (epoch - 1) % config.training.update_lr_epoch_n == 0:
        lr = config.training.lr * config.training.update_lr_
        for param_group in optim.param_groups:
            param_group["lr"] = lr

    y_num = max_n
    for i, sample in enumerate(shared_dataloader):
        img, keypoint, iskpvisible, box_obj, img_label = (
            sample["img"],
            sample["kp"],
            sample["iskpvisible"],
            sample["box_obj"],
            sample["label"],
        )
        # obj_mask = sample["obj_mask"]
        index = sample["y_idx"]

        img = img.cuda()
        keypoint = keypoint.cuda()
        iskpvisible = iskpvisible.cuda()
        # obj_mask = obj_mask.cuda()
        img_label = img_label.cuda()

        # feature is of shape [batch, -1, d_feature (128 as setted)]
        features = net.forward(img, keypoint_positions=keypoint)  # , obj_mask=1 - obj_mask)

        # similarity: [n, k, l]
        if config.training.separate_bank:
            similarity, y_idx, noise_sim, label_onehot = fbank(
                features.to(last_device),
                index.to(last_device),
                iskpvisible.to(last_device),
                img_label.to(last_device),
            )
        else:
            similarity, y_idx, noise_sim, label_onehot = fbank(
                features,
                index.cuda(),
                iskpvisible,
                img_label,
            )

        similarity /= config.training.T

        # make near vertice large value for CE, remove effect of near vertices.
        mask_distance_legal = mask_remove_near(
            keypoint,
            thr=config.training.distance_thr,
            num_neg=config.model.num_noise * config.model.max_group,
            img_label=img_label,
            pad_index=pad_index,
            nb_classes=len(config.dataset.classes),
            zeros=zeros,
            dtype_template=similarity,
            neg_weight=config.training.weight_noise,
        )

        iskpvisible_float = iskpvisible
        iskpvisible = iskpvisible.type(torch.bool).to(iskpvisible.device)

        # Keypoints loss
        loss = criterion(
            (
                similarity.view(-1, similarity.shape[2])
                - mask_distance_legal.view(-1, similarity.shape[2])
            )[
                iskpvisible.view(-1),
                :,
            ],
            y_idx.view(-1)[iskpvisible.view(-1)],
        )

        loss = torch.mean(loss)

        loss_main = loss.item()
        if config.model.num_noise > 0:
            # The loss of noise
            loss_reg = torch.mean(noise_sim) * 0.1
            loss += loss_reg
        else:
            loss_reg = torch.zeros(1)

        loss.backward()
        if iter_num % config.training.accumulate == 0:
            optim.step()
            optim.zero_grad()
            print(
                "n_iter",
                iter_num,
                "epoch",
                epoch,
                "loss",
                "%.5f" % loss_main,
                "loss_reg",
                "%.5f" % loss_reg.item(),
            )
        iter_num += 1

    if (epoch + 1) % 40 == 0:
        save_checkpoint(
            {
                "state": net.state_dict(),
                "memory": fbank.memory,
                "timestamp": int(datetime.timestamp(datetime.now())),
                "args": args,
            },
            "classification_saved_model_%02d.pth" % epoch,
        )
