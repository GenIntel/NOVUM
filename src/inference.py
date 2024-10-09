import BboxTools as bbt
import torch
from tqdm import tqdm
from pathlib import Path
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.utils.data
import torchvision.transforms as transforms
from dataset.Pascal3DPlus import ToTensor, Normalize, Pascal3DPlus
from models.KeypointRepresentationNet import NetE2E
from models.FeatureBank import FeatureBank
from pytorch3d.renderer import (
    RasterizationSettings,
    PerspectiveCameras,
    MeshRasterizer,
)
from lib.MeshUtils import (
    normalize,
    MeshInterpolateModule,
    load_off,
    camera_position_to_spherical_angle,
)
import numpy as np
from lib.get_n_list import get_n_list
from lib.helper import display_results, prepare_rendering_for_init, get_object_texture
from lib.losses import loss_fun, cal_err, cal_rotation_matrix, get_init_pos
torch.cuda.set_device(0)
f = float


from lib.config import load_config, parse_args

args = parse_args()
config = load_config(args, load_default_config=False, log_info=False)

##########################################################################
# General
n_list_set = []
for class_ in config.dataset.classes:
    mesh_path = config.dataset.paths.mesh % class_
    n_list = get_n_list(mesh_path)
    n_list_set.append(n_list[0])
max_n = max(n_list_set)

##########################################################################
# Feature extraction
net = NetE2E(
    net_type=config.model.backbone,
    local_size=[config.model.local_size, config.model.local_size],
    output_dimension=config.model.d_feature,
    n_noise_points=config.model.num_noise,
    pretrain=True,
)
net = torch.nn.DataParallel(net).cuda()
net.eval()

transforms = transforms.Compose([ToTensor(), Normalize()])
criterion = torch.nn.CrossEntropyLoss(reduction="mean").cuda()

checkpoint = torch.load(config.model.ckpt, map_location="cuda:0")
net.load_state_dict(checkpoint["state"], strict=False)
# print net memory usage
total_params = sum(p.numel() for p in net.parameters())
print("Model loaded from " + config.model.ckpt)

##########################################################################
# Load texture objects textures and clutter bank
fbank = FeatureBank(
    inputSize=config.model.d_feature, 
    outputSize=len(config.dataset.classes)*max_n+config.model.num_noise*config.model.max_group, 
    num_pos=len(config.dataset.classes) * max_n,
    num_noise=config.model.num_noise,
    momentum=config.model.adj_momentum,
)
fbank.load_memory(checkpoint["memory"].clone().detach().cpu())


objects_texture = fbank.features
clutter_bank = fbank.clutter
mean_clutter_bank = normalize(torch.mean(clutter_bank, dim=0)).unsqueeze(0).cuda()
##########################################################################
# Rendering
render_image_size = max(config.dataset.image_size) // config.model.down_sample_rate
map_shape = (
    config.dataset.image_size[0] // config.model.down_sample_rate,
    config.dataset.image_size[1] // config.model.down_sample_rate,
)
cameras = PerspectiveCameras(
    focal_length=3000 // config.model.down_sample_rate,
    principal_point=((map_shape[1] // 2, map_shape[0] // 2),),
    image_size=(map_shape,),
    in_ndc=False,
).cuda()
raster_settings = RasterizationSettings(
    image_size=map_shape,
    blur_radius=0.0,
    faces_per_pixel=1,
    bin_size=0,
)
rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

pre_rendered_samples, pre_rendered_maps = prepare_rendering_for_init(config, objects_texture, n_list_set, rasterizer)
##########################################################################
# Inference
Pascal3D_dataset = Pascal3DPlus(
    config=config.dataset,
    transforms=transforms,
    max_n=max_n,
    occlusion=config.inference.occlusion,
    test=True,
)
dataset_size = len(Pascal3D_dataset)
Pascal3D_dataloader = torch.utils.data.DataLoader(
    Pascal3D_dataset,
    batch_size=config.inference.batch_size,
    shuffle=True,
    num_workers=config.workers,
)
class_gds = []
class_preds = []
class_scores = []
pose_errors = []
pose_scores = []
wrong_classes = []
wrong_samples = []
for j, sample in enumerate(tqdm(Pascal3D_dataloader)):
    with torch.no_grad():
        img = sample["img"]
        img = img.cuda()
        keypoint, iskpvisible, this_name, box_obj = (
            sample["kp"],
            sample["iskpvisible"],
            sample["this_name"],
            sample["box"],
        )
        dist_gd, elev_gd, azum_gd, theta_gd = tuple(
            sample["pose"][0],
        )  # 5, elevation, azimuth, theta
        pose_gd = (
            f(theta_gd.item()),
            f(elev_gd.item()),
            f(azum_gd.item()),
            f(dist_gd.item()),
        )
        label = sample["label"].item()
        class_gds.append(label)
        # Features extraction
        predicted_features = net.module.forward_test(img)
        if config.inference.mask_out_padded:
            box_obj = bbt.from_numpy(box_obj.squeeze(0).numpy())
            object_height, object_width = box_obj[0], box_obj[1]
            object_height = (
                object_height[0] // config.model.down_sample_rate,
                object_height[1] // config.model.down_sample_rate,
            )
            object_width = (
                object_width[0] // config.model.down_sample_rate,
                object_width[1] // config.model.down_sample_rate,
            )
            predicted_features = predicted_features[
                ...,
                object_height[0] : object_height[1],
                object_width[0] : object_width[1],
            ]
            img_cropped = img[
                ...,
                object_height[0]
                * config.model.down_sample_rate : object_height[1]
                * config.model.down_sample_rate,
                object_width[0] * config.model.down_sample_rate : object_width[1] * config.model.down_sample_rate,
            ]

        # Classification
        scores = []
        compare_bank = objects_texture.cuda()
        score_per_pixel = compare_bank @ predicted_features.reshape(
            predicted_features.shape[1],
            -1,
        )
        score_per_pixel = score_per_pixel / 2 + 0.5
        scores_val, score_idx = torch.max(score_per_pixel, dim=0)
        
        # For loop 
        for cls_idx, cls in enumerate(config.dataset.classes):
            score_to_keep = (score_idx >= (cls_idx * max_n)) & (score_idx < (cls_idx * max_n + n_list_set[cls_idx]))
            score = torch.sum(scores_val[score_to_keep]) / compare_bank.shape[0]
            scores.append(score.item())
        scores = np.array(scores)
        # apply softmax
        scores = np.exp(scores) / np.sum(np.exp(scores))

        cls_pred = np.argmax(scores)
        class_preds.append(cls_pred)
        class_scores.append(scores)

        # Pose estimation
        # reshape
        if config.inference.use_clutter:
            mclutter_score = mean_clutter_bank @ predicted_features.reshape(
                predicted_features.shape[1],
                -1,
            )
            mclutter_score = torch.sum(mclutter_score, dim=0).reshape(
                predicted_features.shape[-2],
                predicted_features.shape[-1],
            )
        else:
            mclutter_score = None
        C, theta = get_init_pos(
            pre_rendered_samples,
            pre_rendered_maps[label][
                ...,
                object_height[0] : object_height[1],
                object_width[0] : object_width[1],
            ],
            predicted_features,
            config.dataset.distance,
            mclutter_score,
            device="cuda",
        )
        # end torch.no_grad()
    # Pose optimization
    C = torch.nn.Parameter(C, requires_grad=True)
    theta = torch.nn.Parameter(theta, requires_grad=True)

    optim = torch.optim.Adam(
        params=[C, theta],
        lr=config.inference.render_and_compare.lr,
        betas=(config.inference.render_and_compare.adam_beta_0, config.inference.render_and_compare.adam_beta_1),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.2)

    xvert, xface = load_off(config.dataset.paths.mesh % config.dataset.classes[label], to_torch=True)
    inter_module = MeshInterpolateModule(
        xvert.cuda(),
        xface.cuda(),
        get_object_texture(config, objects_texture, label, n_list_set).cuda(),
        rasterizer,
        # post_process=center_crop_fun(map_shape, (render_image_size,) * 2),
    )
    inter_module = inter_module.cuda()
    records = []

    for epoch in range(config.inference.render_and_compare.epochs):
        projected_map = inter_module(C, theta).squeeze()
        projected_map = projected_map[
            ...,
            object_height[0] : object_height[1],
            object_width[0] : object_width[1],
        ]
        object_score = torch.sum(projected_map * predicted_features, dim=0)
        loss = loss_fun(object_score, mclutter_score)
        loss.backward()
        optim.step()
        optim.zero_grad()
        (
            distance_pred,
            elevation_pred,
            azimuth_pred,
        ) = camera_position_to_spherical_angle(C)
        records.append(
            [
                f(theta.item()),
                f(elevation_pred.item()),
                f(azimuth_pred.item()),
                f(distance_pred.item()),
            ],
        )
    pose_pred = tuple(records[-1])
    gd_matrix = cal_rotation_matrix(*pose_gd)
    pred_matrix = cal_rotation_matrix(*pose_pred)
    if (
        np.any(np.isnan(gd_matrix))
        or np.any(np.isnan(pred_matrix))
        or np.any(np.isinf(gd_matrix))
        or np.any(np.isinf(pred_matrix))
    ):
        error_ = np.pi / 2
    else:
        error_ = cal_err(gd_matrix, pred_matrix)
    
    pose_errors.append(error_)

display_results(class_gds, class_preds, pose_errors, thresholds=[np.pi / 6, np.pi / 18])
