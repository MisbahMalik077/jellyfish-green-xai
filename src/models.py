"""Model zoo: ImageNet-pretrained backbones adapted to 1-channel 128x128 gas maps, plus a small CNN."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

MODELS = ["resnet50", "resnet18", "efficientnet_b0", "mobilenetv3_small_100", "vit_tiny_patch16_224", "small_cnn"]
PRETTY = {"resnet50": "ResNet-50", "resnet18": "ResNet-18", "efficientnet_b0": "EfficientNet-B0",
          "mobilenetv3_small_100": "MobileNetV3-S", "vit_tiny_patch16_224": "ViT-Tiny", "small_cnn": "SmallCNN"}


class SmallCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), blk(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build(name: str, pretrained: bool = True) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN()
    kw = dict(pretrained=pretrained, num_classes=2, in_chans=1)  # timm sums RGB weights for 1-channel input
    if name.startswith("vit"):
        kw["img_size"] = 128  # positional embeddings interpolated from 224
    return timm.create_model(name, **kw)


def cam_target_layers(model: nn.Module, name: str):
    """Last convolutional block (or last transformer block for ViT) for CAM methods."""
    if name.startswith("resnet"):
        return [model.layer4[-1]]
    # every CNN uses its last spatial feature map AFTER normalisation + activation (4x4 at 128 px input)
    if name.startswith("efficientnet"):
        return [model.bn2]  # conv_head -> BatchNorm + SiLU
    if name.startswith("mobilenet"):
        return [model.blocks[-1]]  # ConvBnAct (576 ch), already post-activation
    if name.startswith("vit"):
        return [model.blocks[-1].norm1]
    if name == "small_cnn":
        return [model.features[-1][-2]]  # last ReLU before pooling
    raise ValueError(name)


def vit_reshape(tensor, h=8, w=8):
    """Token sequence -> (B, C, H, W) for Grad-CAM on ViT (drops the CLS token)."""
    r = tensor[:, 1:, :].reshape(tensor.size(0), h, w, tensor.size(2))
    return r.permute(0, 3, 1, 2)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model, npix: int = 128) -> int:
    """Forward FLOPs for one image (multiply-add = 2 FLOPs)."""
    model = model.eval().cpu()
    with FlopCounterMode(display=False) as fc:
        model(torch.zeros(1, 1, npix, npix))
    return int(fc.get_total_flops())
