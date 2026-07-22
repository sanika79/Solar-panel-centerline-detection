"""U-Net with a pretrained encoder, via segmentation_models_pytorch.

ResNet34 is the default: fast, well-tested, and reasonable at this data
scale (~530 positive tiles). Encoder is swappable for the ablation
(e.g. a SegFormer/MiT transformer encoder) without touching anything
else in the pipeline.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
