import timm
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from detectron2.modeling.backbone.fpn import FPN
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec
import einops as E
import torch.nn.functional as F

def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
    if output_type == "cls":
        assert cls_token is not None
        output = cls_token
    elif output_type == "gap":
        output = dense_tokens.mean(dim=1)
    elif output_type == "dense":
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        output = dense_tokens.contiguous()
    elif output_type == "dense-cls":
        assert cls_token is not None
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        cls_token = cls_token[:, :, None, None].repeat(1, 1, h, w)
        output = torch.cat((dense_tokens, cls_token), dim=1).contiguous()
    else:
        raise ValueError()

    return output

def center_padding(images, patch_size):
    _, _, h, w = images.shape
    
    # Check if the current dimensions are already evenly divisible by patch_size
    if (h % patch_size == 0 and (h // patch_size) % 2 == 0) and \
       (w % patch_size == 0 and (w // patch_size) % 2 == 0):
        return images  # No padding required

    # Ensure (new_h // patch_size) and (new_w // patch_size) are even
    new_h = ((h // patch_size) + 1) * patch_size
    new_w = ((w // patch_size) + 1) * patch_size

    # If division results in an odd number, adjust to the next even multiple
    if (new_h // patch_size) % 2 != 0:
        new_h += patch_size
    if (new_w // patch_size) % 2 != 0:
        new_w += patch_size

    pad_h = new_h - h
    pad_w = new_w - w

    pad_t = pad_h // 2
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_b = pad_h - pad_t

    images = F.pad(images, (pad_l, pad_r, pad_t, pad_b))
    return images, (pad_l, pad_r, pad_t, pad_b)

def center_crop(output, original_size, pad, down_sample_rate=1):
    padding_to_remove = [int(float(p) * 8 / 14) for p in pad]
    (pad_l, pad_r, pad_t, pad_b) = padding_to_remove
    output = output[..., pad_t:output.shape[2] - pad_b, pad_l:output.shape[3] - pad_r]
    # resize to original_size / down_sample_rate
    output = interpolate(output, size=(original_size[0] // down_sample_rate, original_size[1] // down_sample_rate), mode='bilinear', align_corners=False)
    return output

class ViTBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
        }
        self.type_ = cfg.backbone.type
        self.arch = cfg.backbone.arch
        self.output = cfg.backbone.output
        self.down_sample_rate = cfg.down_sample_rate
        assert self.output in ["cls", "gap", "dense", "dense-cls"]
        
        self.vit = ViTBackbone.create_model(model_type=self.type_, arch=self.arch)
        self.has_registers = "_reg" in self.arch
        
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]
        
        feat_dim = feat_dims[self.arch]
        feat_dim = feat_dim * 2 if self.output == "dense-cls" else feat_dim
        

        num_layers = len(self.vit.blocks)
        self.multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]
        self.feat_dims = [feat_dim]*len(self.multilayers)
        
        # Remove classification head
        self.vit.head = nn.Identity()
        
        # self._out_features = ['p2', 'p3', 'p4', 'p5', 'p6']
        # self._out_feature_channels = {'p2': feat_dim, 'p3': feat_dim, 'p4': feat_dim, 'p5': feat_dim, 'p6': feat_dim}
        
    @staticmethod
    def create_model(model_type: str, arch: str) -> nn.Module:
        """
        Args:
            model_type (str): Type of ViT model (e.g., "dino", "dinov2", "vanilla").
            arch (str): Architecture of ViT (e.g., "vitb16", "vitl14").
        :return: the model
        """
        if 'dino' in model_type:
            model = torch.hub.load(f'facebookresearch/{model_type}:main', f'{model_type}_{arch}')
        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            timm_models = {
                "vitb8": "vit_base_patch8_224",
                "vitb16": "vit_base_patch16_224",
                "vitb14": "vit_base_patch14_224",
                "vitl14": "vit_large_patch14_224",
                "vitg14": "vit_giant_patch14_224",
            }
            temp_model = timm.create_model(timm_models[arch], pretrained=True)
            model = torch.hub.load('facebookresearch/dino:main', f'dino_{arch}')
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict['head.weight']
            del temp_state_dict['head.bias']
            model.load_state_dict(temp_state_dict)
            
        return model.eval().to(torch.float32)
    
    def forward(self, images):
        """
        Forward pass for ViT with multi-layer feature extraction.

        Args:
            images (torch.Tensor): Input images of shape (B, C, H, W)

        Returns:
            dict: Feature maps mapped to self._out_features
        """

        # Pad images to ensure dimensions match patch size
        images, pad = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size
        # Prepare token embeddings
        if self.type_ == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        # Extract features from selected layers
        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):  # Stop when all required layers are collected
                    break

        # Convert token embeddings into feature maps
        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]  # Extract CLS token
            spatial = x_i[:, -1 * num_spatial :]  # Extract patch embeddings

            # Convert tokens to structured output
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            # Map feature output to `_out_features`
            outputs.append(x_i)

        return outputs, pad

class HeadViTBackbone(ViTBackbone):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        # Freeze the ViT Backbone
        if cfg.backbone.freeze:
            for param in self.parameters():
                param.requires_grad = False  # Freeze backbone
        self.image_size = (input_shape.height, input_shape.width)
        self.down_sample_rate = cfg.down_sample_rate
        self.head = DPT(self.feat_dims, output_dim=256, hidden_dim=512, kernel_size=3, last_upscaling=cfg.backbone.last_upscaling)

        # Output features configuration
        self._out_features = ['p2', 'p3', 'p4', 'p5']
        self._out_feature_strides = {'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32}
        self._out_feature_channels = {'p2': 256, 'p3': 256, 'p4': 256, 'p5': 256}

    def forward(self, images):
        # Forward pass through frozen backbone
        with torch.no_grad():  # Prevents gradient computation for backbone
            features, pad = super().forward(images)
        # Forward pass through FPN (trainable)
        output = self.head(features)
        # center crop to remove padding
        return center_crop(output["p2"], self.image_size, pad, self.down_sample_rate)
    
@BACKBONE_REGISTRY.register()
def build_vit(cfg, input_shape: ShapeSpec) -> Backbone:
    return HeadViTBackbone(cfg, input_shape)




class FeatureFusionBlock(nn.Module):
    def __init__(self, features, kernel_size, with_skip=True):
        super().__init__()
        self.with_skip = with_skip
        if self.with_skip:
            self.resConfUnit1 = ResidualConvUnit(features, kernel_size)

        self.resConfUnit2 = ResidualConvUnit(features, kernel_size)

    def forward(self, x, skip_x=None):
        if skip_x is not None:
            assert self.with_skip and skip_x.shape == x.shape
            x = self.resConfUnit1(x) + skip_x

        x = self.resConfUnit2(x)
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size needs to be odd"
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x) + x


class DPT(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=3, last_upscaling=True):
        super().__init__()
        self.last_upscaling = last_upscaling
        assert len(input_dims) == 4
        self.conv_0 = nn.Conv2d(input_dims[0], hidden_dim, 1, padding=0)
        self.conv_1 = nn.Conv2d(input_dims[1], hidden_dim, 1, padding=0)
        self.conv_2 = nn.Conv2d(input_dims[2], hidden_dim, 1, padding=0)
        self.conv_3 = nn.Conv2d(input_dims[3], hidden_dim, 1, padding=0)

        self.ref_0 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_1 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_2 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_3 = FeatureFusionBlock(hidden_dim, kernel_size, with_skip=False)

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

    def forward(self, feats):
        """Prediction each pixel."""
        assert len(feats) == 4
        feats[0] = self.conv_0(feats[0])
        feats[1] = self.conv_1(feats[1])
        feats[2] = self.conv_2(feats[2])
        feats[3] = self.conv_3(feats[3])

        feats = [interpolate(x, scale_factor=2, mode="nearest") for x in feats]
        out = self.ref_3(feats[3], None)
        out = self.ref_2(feats[2], out)
        out = self.ref_1(feats[1], out)
        out = self.ref_0(feats[0], out)
        out = interpolate(out, scale_factor=4)
        out = self.out_conv(out)
        
        if self.last_upscaling:
            out = interpolate(out, scale_factor=2, mode="nearest")
        return {"p2":out}

def make_conv(input_dim, hidden_dim, output_dim, num_layers, kernel_size=1):
    if num_layers == 1:
        conv = nn.Conv2d(input_dim, output_dim, kernel_size)
    else:
        assert num_layers > 1
        modules = [nn.Conv2d(input_dim, hidden_dim, kernel_size), nn.ReLU(inplace=True)]
        for i in range(num_layers - 2):
            modules.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size))
            modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv2d(hidden_dim, output_dim, kernel_size))
        conv = nn.Sequential(*modules)

    return conv


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1):
        super().__init__()
        if type(input_dim) is not int:
            input_dim = sum(input_dim)

        assert type(input_dim) is int
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding)

    def forward(self, feats):
        if type(feats) is list:
            feats = torch.cat(feats, dim=1)

        feats = interpolate(feats, scale_factor=4, mode="bilinear")
        return self.conv(feats)


class MultiscaleHead(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=1):
        super().__init__()

        self.convs = nn.ModuleList(
            [make_conv(in_d, None, hidden_dim, 1, kernel_size) for in_d in input_dims]
        )
        interm_dim = len(input_dims) * hidden_dim
        self.conv_mid = make_conv(interm_dim, hidden_dim, hidden_dim, 3, kernel_size)
        self.conv_out = make_conv(hidden_dim, hidden_dim, output_dim, 2, kernel_size)

    def forward(self, feats):
        num_feats = len(feats)
        feats = [self.convs[i](feats[i]) for i in range(num_feats)]

        h, w = feats[-1].shape[-2:]
        feats = [interpolate(feat, (h, w), mode="bilinear") for feat in feats]
        feats = torch.cat(feats, dim=1).relu()

        # upsample
        feats = interpolate(feats, scale_factor=2, mode="bilinear")
        feats = self.conv_mid(feats).relu()
        feats = interpolate(feats, scale_factor=4, mode="bilinear")
        return self.conv_out(feats)
    
