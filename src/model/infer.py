"""Load a trained checkpoint and run inference on tiles."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.augment import normalize_imagenet
from data.tile_io import load_tile
from model.unet import build_model


def load_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(encoder_name=ckpt["encoder"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_tile(
    model: torch.nn.Module, image_path: str | Path, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (prob_mask, valid_mask), both (H, W) float/bool arrays."""
    image, valid_mask = load_tile(image_path)
    x = torch.from_numpy(normalize_imagenet(image)).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return prob, valid_mask
