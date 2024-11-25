
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def compute_weighted_correspondances(activation_maps: dict, img_, img_name_):
    class_of_interest = list(activation_maps.keys())

    fig, axes = plt.subplots(1, len(class_of_interest) + 1, figsize=(30, 15))
    # unnormalize image  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    img_unnorm = img_ * torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda(img_.device) + torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda(img_.device)
    axes[0].imshow(img_unnorm.squeeze().permute(1, 2, 0).cpu().numpy())
    axes[0].axis("off")
    axes[0].set_title(f"{img_name_}")

    for class_index, current_cls in enumerate(class_of_interest):
        current_actmap = activation_maps[current_cls].clone()
        current_actmap = (current_actmap.detach().cpu().numpy()*255).astype(np.uint8)
        current_actmap = cv2.cvtColor(current_actmap, cv2.COLOR_BGR2RGB)
        axes[class_index + 1].imshow(current_actmap)
        axes[class_index + 1].set_title(f"{current_cls}")
        axes[class_index + 1].axis("off")
    fig.canvas.draw()
    img_to_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_to_plot = img_to_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img_to_plot
