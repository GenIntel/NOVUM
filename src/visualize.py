
import numpy as np
from lib.get_n_list import get_n_list
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import cv2
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.utils.data
import torchvision.transforms as trans
from dataset.Pascal3DPlus import ToTensor, Normalize, Pascal3DPlus
from lib.visualization import compute_weighted_correspondances
from models.KeypointRepresentationNet import NetE2E
from pytorch3d.renderer import (
    RasterizationSettings,
    PerspectiveCameras,
    MeshRasterizer,
)
from lib.MeshUtils import normalize
from lib.config import load_config, parse_args


device_str = "cuda:0"

args = parse_args()
config = load_config(args, load_default_config=False, log_info=False)

#####
# PARAMS TO SET
viz_path = "PATH/TO/VIZ"
occ_level = ""
clutter_th = 0.65  # Threshold for clutter
#####

# Cuboid with gradient color texture (used for visualization)
texture_color_bank = torch.load(Path(viz_path, "texture_bank.pth"), map_location="cuda:0")

assert occ_level in config.dataset.occlusion_levels, "Invalid occ_level"

if occ_level:
    dataroot = str(Path(config.dataset.paths.root, config.dataset.paths.eval_iid))
else:
    dataroot = str(Path(config.dataset.paths.root, config.dataset.paths.eval_ood))

mesh_path_ref = str(Path(config.dataset.paths.root, config.dataset.paths.mesh))
classification_size = (int(config.dataset.image_size[0]), int(config.dataset.image_size[1]))

net = NetE2E(
    net_type="resnetext",
    local_size=(config.model.local_size, config.model.local_size),
    output_dimension=config.model.d_feature,
    n_noise_points=config.model.num_noise,
    pretrain=True,
)
net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
net.eval()

transforms = trans.Compose([ToTensor(), Normalize()])
criterion = torch.nn.CrossEntropyLoss(reduction="mean").cuda()

checkpoint = torch.load(config.model.ckpt, map_location="cuda:0")
incompatible_keys = net.load_state_dict(checkpoint["state"], strict=False)
print("Keys not found:", incompatible_keys)
# print net memory usage
total_params = sum(p.numel() for p in net.parameters())
print("Model loaded from " + config.model.ckpt)

n_list_set = []
for class_ in config.dataset.classes:
    mesh_path = Path(config.dataset.paths.root, config.dataset.paths.mesh, class_)
    n_list = get_n_list(mesh_path)
    n_list_set.append(n_list[0])
max_n = max(n_list_set)

##########################################################################
# Load texture objects textures and clutter bank
objects_texture_size = len(config.dataset.classes) * max_n
clutter_bank_size = config.model.num_noise * config.model.max_group
memory_size = objects_texture_size + clutter_bank_size
objects_texture = checkpoint["memory"][0:objects_texture_size].clone().detach().cpu()
clutter_bank = checkpoint["memory"][objects_texture_size:].clone().detach().cpu()
mean_clutter_bank = normalize(torch.mean(clutter_bank, dim=0)).unsqueeze(0).cuda()
##########################################################################
# Rendering
down_sample_rate = 8
set_distance = 5.0
render_image_size = max(classification_size) // down_sample_rate
map_shape = (
    classification_size[0] // down_sample_rate,
    classification_size[1] // down_sample_rate,
)
cameras = PerspectiveCameras(
    focal_length=3000 // down_sample_rate,
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

#########################################################################
with open(Path(viz_path, "file_list.txt"), "r") as f:
    file_list = f.readlines()
file_list = [x.strip() for x in file_list]  
with open(Path(viz_path, "label_list.txt"), "r") as f:
    label_list = f.readlines()
label_list = [int(x.strip()) for x in label_list]  
Pascal3D_dataset = Pascal3DPlus(
    config=config.dataset,
    transforms=transforms,
    max_n=max_n,
    occlusion=occ_level,
    test=True,
)
Pascal3D_dataset.file_list = file_list
Pascal3D_dataset.label_list = label_list
dataset_size = len(Pascal3D_dataset)
print(f"Dataset size: {dataset_size}")
Pascal3D_dataloader = torch.utils.data.DataLoader(
    Pascal3D_dataset,
    batch_size=config.inference.batch_size,
    shuffle=True,
    num_workers=1,
)

for i, sample in enumerate(tqdm(Pascal3D_dataloader)):
    img, keypoint, iskpvisible, box_obj, img_label = (
        sample["img"],
        sample["kp"],
        sample["iskpvisible"],
        sample["box_obj"],
        sample["label"],
    )

    obj_mask = sample["obj_mask"]
    index = sample["y_idx"]

    img = img.to(device_str)
    keypoint = keypoint.to(device_str)
    iskpvisible = iskpvisible.squeeze(0).bool().to(device_str)
    obj_mask = obj_mask.to(device_str)
    index = index.to(device_str).squeeze(0)
    img_label = img_label.to(device_str)
    cls_name = config.dataset.classes[img_label.item()]
    img_name = cls_name + "/" + sample["this_name"][0]

    features = net.module.forward_test(img)
    
    compare_bank = checkpoint["memory"][0 : (len(config.dataset.classes) * max_n)]
    weighted_nocs = {}
    # get 2 random classes
    all_cls = config.dataset.classes.copy()
    all_cls.remove(cls_name)
    cls_of_interest = [cls_name] + np.random.choice(all_cls, 2, replace=False).tolist()
    for current_cls in [config.dataset.classes.index(cls_) for cls_ in cls_of_interest]:
        score_per_pixel = compare_bank[current_cls * max_n:(current_cls + 1) * max_n] @ features.reshape(
            features.shape[1],
            -1,
        )
        score_per_pixel = score_per_pixel / 2 + 0.5
        scores_val, score_idx = torch.max(score_per_pixel, dim=0)
        output_activation_nocs = torch.zeros(scores_val.shape[0], 3).to(texture_color_bank.device)
        score_idx_2d_map = score_idx.clone().detach()
        score_idx_2d_map[scores_val < clutter_th] = 0
        non_zero_idx_2d = score_idx_2d_map != 0
        max_ = 1.
        min_ = scores_val[non_zero_idx_2d].min()
        score_val_norm = (scores_val[non_zero_idx_2d].unsqueeze(1) - min_ + 0.15) / (max_ - min_ + 0.15)
        output_activation_nocs[non_zero_idx_2d] = texture_color_bank[current_cls * max_n + score_idx_2d_map[non_zero_idx_2d]] * score_val_norm
        output_activation_nocs = output_activation_nocs.reshape(features.shape[2], features.shape[3], 3)        

        weighted_nocs[config.dataset.classes[current_cls]] = output_activation_nocs
    img_to_plot = compute_weighted_correspondances(
        weighted_nocs, img[0], img_name
    )
    # save image
    img_to_save = Image.fromarray(img_to_plot)
    path = Path(viz_path, "output", f"{img_name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    img_to_save.save(path)
