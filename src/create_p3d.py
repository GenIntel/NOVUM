import numpy as np
import BboxTools as bbt
import scipy.io as sio
from pathlib import Path
import os
import gdown
from PIL import Image
import pickle
import cv2
import math
from lib.config import load_config, parse_args

mesh_len = {'aeroplane': 8, 'bicycle': 6, 'boat': 6, 'bottle': 8, 'bus': 6, 'car': 10, 'chair': 10, 'diningtable': 6, 'motorbike': 5, 'sofa': 6, 'train': 4, 'tvmonitor': 4}

args = parse_args()
config = load_config(args, load_default_config=True, log_info=False)

project_dir = Path(__file__).resolve().parent
root = Path(config.dataset.paths.root)
# download data
root.mkdir(parents=True, exist_ok=True)
mesh_para_names = config.dataset.required_annotations
dataset_root = root / "PASCAL3D+_release1.1"
occluded_dataset_root = root / "OccludedPASCAL3D"

# download raw dataset 
if not dataset_root.exists():
    print("Downloading Pascal3D+ dataset (1/2)")
    os.system("cd " + str(root) + " && " + "wget ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip" + " && " + "unzip PASCAL3D+_release1.1.zip" + " && " + "rm PASCAL3D+_release1.1.zip" + " && " + f"cd " + str(project_dir))

# download subsets 
if not (dataset_root / "Image_subsets").exists():
    print("Downloading Pascal3D+ dataset (2/2)")
    gdown.download("https://docs.google.com/uc?export=download&id=1NsoVXW8ngQCqTHHFSW8YYsCim9EjiXS7", dataset_root / "Image_subsets.zip", quiet=False)
    os.system("unzip " + str(dataset_root / "Image_subsets.zip") + " -d " + str(dataset_root) + " && " + "rm " + str(dataset_root / "Image_subsets.zip"))

# download occluded dataset
if not occluded_dataset_root.exists():
    print("Downloading Occluded Pascal3D+ dataset (1/1)")
    occluded_dataset_root.mkdir(parents=True, exist_ok=True)
    gdown.download("https://docs.google.com/uc?export=download&id=1X-xwyypLTm9vr-boLYPIPhGxcYaPHSNF", str(occluded_dataset_root / "OccludedPASCAL3D_FGL1_BGL1.zip"), quiet=False)
    gdown.download("https://docs.google.com/uc?export=download&id=1dNP8YE3RJ9Pzr_jQ11O6f6eYgYnq9ROp", str(occluded_dataset_root / "OccludedPASCAL3D_FGL2_BGL2.zip"), quiet=False)
    gdown.download("https://docs.google.com/uc?export=download&id=1GsHCyAYnqcJsAgiih1vKpDQxzF3ouFxS", str(occluded_dataset_root / "OccludedPASCAL3D_FGL3_BGL3.zip"), quiet=False)
    os.system("unzip " + str(occluded_dataset_root / "OccludedPASCAL3D_FGL1_BGL1.zip") + " -d " + str(occluded_dataset_root) + " && " + "rm " + str(occluded_dataset_root / "OccludedPASCAL3D_FGL1_BGL1.zip"))
    os.system("unzip " + str(occluded_dataset_root / "OccludedPASCAL3D_FGL2_BGL2.zip") + " -d " + str(occluded_dataset_root) + " && " + "rm " + str(occluded_dataset_root / "OccludedPASCAL3D_FGL2_BGL2.zip"))
    os.system("unzip " + str(occluded_dataset_root / "OccludedPASCAL3D_FGL3_BGL3.zip") + " -d " + str(occluded_dataset_root) + " && " + "rm " + str(occluded_dataset_root / "OccludedPASCAL3D_FGL3_BGL3.zip"))
    

def get_anno(record, *args, idx=0):
    out = []
    for key_ in args:
        if key_ == "height":
            out.append(record["imgsize"][0, 0][0][1])
        elif key_ == "width":
            out.append(record["imgsize"][0, 0][0][0])
        elif key_ == "bbox":
            out.append(record["objects"][0, 0]["bbox"][0, idx][0])
        elif key_ == "cad_index":
            out.append(record["objects"][0, 0]["cad_index"][0, idx][0, 0])
        elif key_ == "principal":
            px = record["objects"][0, 0]["viewpoint"][0, idx]["px"][0, 0][0, 0]
            py = record["objects"][0, 0]["viewpoint"][0, idx]["py"][0, 0][0, 0]
            out.append(np.array([px, py]))
        elif key_ in ["theta", "azimuth", "elevation"]:
            out.append(
                record["objects"][0, 0]["viewpoint"][0, idx][key_][0, 0][0, 0]
                * math.pi
                / 180
            )
        else:
            out.append(record["objects"][0, 0]["viewpoint"][0, idx][key_][0, 0][0, 0])

    if len(out) == 1:
        return out[0]

    return tuple(out)


DATASET_SUBSET = "imagenet"
IMAGE_SIZE = config.dataset.image_size  # [H, W]
# Training
train_root = Path(config.dataset.paths.root, config.dataset.paths.training)
print("Creating training dataset: ")
for category in config.dataset.classes:
    out_shape = [
        ((IMAGE_SIZE[0] - 1) // 32 + 1) * 32,
        ((IMAGE_SIZE[1] - 1) // 32 + 1) * 32,
    ]
    out_shape = [int(out_shape[0]), int(out_shape[1])]
    # Kp_list
    save_image_path = train_root / "images" / f"{category}"
    save_annotation_path = train_root / "annotations" / f"{category}"
    save_list_path = train_root / "lists" / f"{category}"
    save_image_path.mkdir(parents=True, exist_ok=True)
    save_annotation_path.mkdir(parents=True, exist_ok=True)
    save_list_path.mkdir(parents=True, exist_ok=True)
    # Path
    list_dir = dataset_root / "Image_sets"
    pkl_dir = dataset_root / "Image_subsets"
    anno_dir = dataset_root / "Annotations" / f"{category}_{DATASET_SUBSET}"
    load_image_path = dataset_root / "Images" / f"{category}_{DATASET_SUBSET}"

    file_dir = list_dir / f"{category}_{DATASET_SUBSET}_train.txt"
    with open(file_dir, "r") as fh:
        image_names = fh.readlines()
    image_names = [e.strip() for e in image_names if e != "\n"]

    subtype_file_dir = list_dir / f"{category}_{DATASET_SUBSET}_subtype.txt"
    with open(subtype_file_dir, "r") as fh:
        subtype_list = fh.readlines()
    subtype_list = [e.strip() for e in subtype_list if e != "\n"]
    pkl_path = pkl_dir / f"{category}_{DATASET_SUBSET}_train.pkl"
    subtype_images = pickle.load(open(pkl_path, "rb"))
    annotations = [[] for _ in range(len(subtype_list))]

    mesh_name_list = ["" for _ in range(mesh_len[category])]
    for i in range(len(subtype_list)):
        name_list = ""
        for img_name in subtype_images[i]:
            if not os.path.exists(
                os.path.join(load_image_path, img_name + ".JPEG")
            ):
                continue
            name_list += img_name + ".JPEG\n"
            anno_path = anno_dir / f"{img_name}.mat"
            mat_contents = sio.loadmat(anno_path)
            record = mat_contents["record"]

            mesh_idx = get_anno(record, "cad_index")
            mesh_name_list[mesh_idx - 1] += img_name + ".JPEG\n"
            if (
                (not config.overwrite)
                and (save_annotation_path / f"{img_name}.npz").exists()
                and (save_image_path / f"{img_name}.JPEG").exists()
            ):
                continue
            objects = record["objects"]
            azimuth_coarse = objects[0, 0]["viewpoint"][0, 0]["azimuth_coarse"][
                0, 0
            ][0, 0]
            elevation_coarse = objects[0, 0]["viewpoint"][0, 0]["elevation_coarse"][
                0, 0
            ][0, 0]
            distance = objects[0, 0]["viewpoint"][0, 0]["distance"][0, 0][0, 0]
            bbox = objects[0, 0]["bbox"][0, 0][0]
            box = bbt.from_numpy(bbox, sorts=("x0", "y0", "x1", "y1"))
            resize_rate = float(200 * get_anno(record, "distance") / 1000)
            if resize_rate <= 0:
                resize_rate = 224 / min(box.shape)
            box_ori = box.copy()
            box *= resize_rate
            img = np.array(Image.open(load_image_path / f"{img_name}.JPEG"))
            box_ori = box_ori.set_boundary(img.shape[0:2])
            w, h = img.shape[1], img.shape[0]
            img = cv2.resize(
                img, dsize=(int(w * resize_rate), int(h * resize_rate))
            )
            center = (get_anno(record, "principal")[::-1] * resize_rate).astype(int)
            box1 = bbt.box_by_shape(out_shape, center)
            if (
                out_shape[0] // 2 - center[0] > 0
                or out_shape[1] // 2 - center[1] > 0
                or out_shape[0] // 2 + center[0] - img.shape[0] > 0
                or out_shape[1] // 2 + center[1] - img.shape[1] > 0
            ):
                padding = (
                    (
                        max(out_shape[0] // 2 - center[0], 0),
                        max(out_shape[0] // 2 + center[0] - img.shape[0], 0),
                    ),
                    (
                        max(out_shape[1] // 2 - center[1], 0),
                        max(out_shape[1] // 2 + center[1] - img.shape[1], 0),
                    ),
                )
                mask_padding = padding
                if len(img.shape) == 3:
                    padding += ((0, 0),)

                img = np.pad(img, padding, mode="constant")
                box = box.shift([padding[0][0], padding[1][0]])
                box1 = box1.shift([padding[0][0], padding[1][0]])
            box_in_cropped = box.copy()
            box = box1.set_boundary(img.shape[0:2])
            box_in_cropped = box.box_in_box(box_in_cropped)
            img_cropped = box.apply(img)
            proj_foo = bbt.projection_function_by_boxes(
                box_ori, box_in_cropped, compose=False
            )

            save_parameters = dict(
                name=img_name,
                box=box.numpy(),
                box_ori=box_ori.numpy(),
                box_obj=box_in_cropped.numpy(),
                occ_mask=None,
                cropped_occ_mask=None,
                padding=padding
            )

            save_parameters = {
                **save_parameters,
                **{
                    k: v
                    for k, v in zip(
                        mesh_para_names, get_anno(record, *mesh_para_names)
                    )
                },
            }

            np.savez(
                os.path.join(save_annotation_path, img_name), **save_parameters
            )
            Image.fromarray(img_cropped).save(
                os.path.join(save_image_path, img_name + ".JPEG")
            )

        with open(
            os.path.join(save_list_path, subtype_list[i] + ".txt"), "w"
        ) as fl:
            fl.write(name_list)

    for i, t_ in enumerate(mesh_name_list):
        with open(
            os.path.join(save_list_path, "mesh%02d" % (i + 1) + ".txt"), "w"
        ) as fl:
            fl.write(t_)
print(f"Processed training")

# Evaluation
for occlusion in config.dataset.occlusion_levels:
    save_path_val = Path(
        config.dataset.paths.root, 
        config.dataset.paths.eval_ood if occlusion else config.dataset.paths.eval_iid
    )
    print(f"Creating evaluation dataset (occlusion lvl : {occlusion}): ")
    for category in config.dataset.classes:
        out_shape = [
            ((IMAGE_SIZE[0] - 1) // 32 + 1) * 32,
            ((IMAGE_SIZE[1] - 1) // 32 + 1) * 32,
        ]
        out_shape = [int(out_shape[0]), int(out_shape[1])]
        # Kp_list
        save_image_path = save_path_val / "images" / f"{category}{occlusion}"
        save_annotation_path = save_path_val / "annotations" / f"{category}{occlusion}"
        save_list_path = save_path_val / "lists" / f"{category}{occlusion}"
        save_image_path.mkdir(parents=True, exist_ok=True)
        save_annotation_path.mkdir(parents=True, exist_ok=True)
        save_list_path.mkdir(parents=True, exist_ok=True)
        # Path
        list_dir = dataset_root / "Image_sets"
        pkl_dir = dataset_root / "Image_subsets"
        anno_dir = dataset_root / "Annotations" / f"{category}_{DATASET_SUBSET}"
        load_image_path = dataset_root / "Images" / f"{category}_{DATASET_SUBSET}"
        
        if occlusion:
            load_image_path = occluded_dataset_root / "images" / f"{category}{occlusion}"
            occ_mask_dir = occluded_dataset_root / "annotations" / f"{category}{occlusion}"
        else:
            load_image_path = dataset_root / "Images" / f"{category}_{DATASET_SUBSET}"
            occ_mask_dir = None

        file_dir = list_dir / f"{category}_{DATASET_SUBSET}_val.txt"
        with open(file_dir, "r") as fh:
            image_names = fh.readlines()
        image_names = [e.strip() for e in image_names if e != "\n"]

        subtype_file_dir = list_dir / f"{category}_{DATASET_SUBSET}_subtype.txt"
        with open(subtype_file_dir, "r") as fh:
            subtype_list = fh.readlines()
        subtype_list = [e.strip() for e in subtype_list if e != "\n"]
        pkl_path = pkl_dir, f"{category}_{DATASET_SUBSET}_val.pkl"
        subtype_images = pickle.load(open(pkl_path, "rb"))
        annotations = [[] for _ in range(len(subtype_list))]

        mesh_name_list = ["" for _ in range(mesh_len[category])]
        for i in range(len(subtype_list)):
            name_list = ""
            for img_name in subtype_images[i]:
                if not (load_image_path / f"{img_name}.JPEG").exists():
                    continue
                if occlusion:
                    occ_mask = np.load(occ_mask_dir / f"{img_name}.npz", allow_pickle=True)["occluder_mask"]
                else:
                    occ_mask = None
                name_list += img_name + ".JPEG\n"
                anno_path = anno_dir / f"{img_name}.mat"
                mat_contents = sio.loadmat(anno_path)
                record = mat_contents["record"]

                mesh_idx = get_anno(record, "cad_index")
                mesh_name_list[mesh_idx - 1] += img_name + ".JPEG\n"

                if (
                    (not config.overwrite)
                    and (save_annotation_path / f"{img_name}.npz").exists()
                    and (save_image_path / f"{img_name}.JPEG").exists()
                ):
                    continue
                objects = record["objects"]
                azimuth_coarse = objects[0, 0]["viewpoint"][0, 0]["azimuth_coarse"][
                    0, 0
                ][0, 0]
                elevation_coarse = objects[0, 0]["viewpoint"][0, 0]["elevation_coarse"][
                    0, 0
                ][0, 0]
                distance = objects[0, 0]["viewpoint"][0, 0]["distance"][0, 0][0, 0]
                bbox = objects[0, 0]["bbox"][0, 0][0]
                box = bbt.from_numpy(bbox, sorts=("x0", "y0", "x1", "y1"))
                resize_rate = float(200 * get_anno(record, "distance") / 1000)
                if resize_rate <= 0:
                    resize_rate = 224 / min(box.shape)
                box_ori = box.copy()
                box *= resize_rate
                img = np.array(Image.open(load_image_path / f"{img_name}.JPEG"))
                box_ori = box_ori.set_boundary(img.shape[0:2])
                w, h = img.shape[1], img.shape[0]
                if occlusion:
                    int_occ_mask = occ_mask.astype(int)
                    int_occ_mask = cv2.resize(
                        int_occ_mask,
                        dsize=(int(w * resize_rate), int(h * resize_rate)),
                        interpolation=cv2.INTER_NEAREST,
                    )
                img = cv2.resize(
                    img, dsize=(int(w * resize_rate), int(h * resize_rate))
                )
                center = (get_anno(record, "principal")[::-1] * resize_rate).astype(int)
                box1 = bbt.box_by_shape(out_shape, center)
                if (
                    out_shape[0] // 2 - center[0] > 0
                    or out_shape[1] // 2 - center[1] > 0
                    or out_shape[0] // 2 + center[0] - img.shape[0] > 0
                    or out_shape[1] // 2 + center[1] - img.shape[1] > 0
                ):
                    padding = (
                        (
                            max(out_shape[0] // 2 - center[0], 0),
                            max(out_shape[0] // 2 + center[0] - img.shape[0], 0),
                        ),
                        (
                            max(out_shape[1] // 2 - center[1], 0),
                            max(out_shape[1] // 2 + center[1] - img.shape[1], 0),
                        ),
                    )
                    mask_padding = padding
                    if len(img.shape) == 3:
                        padding += ((0, 0),)
                    img = np.pad(img, padding, mode="constant")
                    if occlusion:
                        int_occ_mask = np.pad(
                            int_occ_mask, mask_padding, mode="constant"
                        )
                    box = box.shift([padding[0][0], padding[1][0]])
                    box1 = box1.shift([padding[0][0], padding[1][0]])
                box_in_cropped = box.copy()
                box = box1.set_boundary(img.shape[0:2])
                box_in_cropped = box.box_in_box(box_in_cropped)
                img_cropped = box.apply(img)
                if occlusion:
                    mask_cropped = box.apply(int_occ_mask)
                else:
                    mask_cropped = None
                proj_foo = bbt.projection_function_by_boxes(
                    box_ori, box_in_cropped, compose=False
                )

                save_parameters = dict(
                    name=img_name,
                    box=box.numpy(),
                    box_ori=box_ori.numpy(),
                    box_obj=box_in_cropped.numpy(),
                    occ_mask=occ_mask,
                    cropped_occ_mask=mask_cropped,
                    padding=padding
                )

                save_parameters = {
                    **save_parameters,
                    **{
                        k: v
                        for k, v in zip(
                            mesh_para_names, get_anno(record, *mesh_para_names)
                        )
                    },
                }

                np.savez(
                    os.path.join(save_annotation_path, img_name), **save_parameters
                )
                Image.fromarray(img_cropped).save(
                    os.path.join(save_image_path, img_name + ".JPEG")
                )

            with open(
                os.path.join(save_list_path, subtype_list[i] + ".txt"), "w"
            ) as fl:
                fl.write(name_list)

        for i, t_ in enumerate(mesh_name_list):
            with open(
                os.path.join(save_list_path, "mesh%02d" % (i + 1) + ".txt"), "w"
            ) as fl:
                fl.write(t_)
    print(f"Processed images with occlusion level: {occlusion}")
