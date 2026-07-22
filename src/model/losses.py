"""Composite BCE + Dice + clDice loss for thin curvilinear segmentation.

clDice (Shit et al., CVPR 2021, "clDice -- A Novel Topology-Preserving
Loss Function for Tubular Structure Segmentation") is included because
plain Dice tends to erode thin (1-5px-wide) targets rather than penalize
breaks in them -- it rewards keeping the *skeleton* of prediction and
target each covered by the other mask, which is what actually matters
for a centerline: staying connected end-to-end, not just overlapping by
area. Implemented via the paper's soft, differentiable morphological
skeletonization (iterated soft-erode/open), not a hard skimage skeleton,
so it's usable as a training loss.

All terms respect `valid_mask` so the synthetic edge-padding on the 23
boundary tiles never contributes to the loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, 3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """Iterative morphological soft-skeleton (Shit et al. 2021, Algorithm 1)."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iterations):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(pred: torch.Tensor, target: torch.Tensor, iterations: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """1 - clDice. pred, target: (B,1,H,W), values in [0,1]."""
    skel_pred = soft_skeletonize(pred, iterations)
    skel_target = soft_skeletonize(target, iterations)

    t_prec = (torch.sum(skel_pred * target) + eps) / (torch.sum(skel_pred) + eps)
    t_sens = (torch.sum(skel_target * pred) + eps) / (torch.sum(skel_target) + eps)
    cl_dice = 2 * t_prec * t_sens / (t_prec + t_sens + eps)
    return 1 - cl_dice


def masked_bce(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = loss * valid_mask
    return loss.sum() / (valid_mask.sum() + eps)


def masked_dice(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred * valid_mask
    target = target * valid_mask
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


LOSS_PRESETS = {
    "bce": dict(bce_weight=1.0, dice_weight=0.0, cldice_weight=0.0),
    "bce_dice": dict(bce_weight=1.0, dice_weight=1.0, cldice_weight=0.0),
    "bce_dice_cldice": dict(bce_weight=1.0, dice_weight=1.0, cldice_weight=1.0),
}


class BCEDiceClDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        cldice_weight: float = 1.0,
        cldice_iterations: int = 10,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.cldice_iterations = cldice_iterations

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred = torch.sigmoid(logits)
        bce = masked_bce(logits, target, valid_mask)
        dice = masked_dice(pred, target, valid_mask)

        if self.cldice_weight > 0:
            cldice = soft_cldice(pred * valid_mask, target * valid_mask, iterations=self.cldice_iterations)
        else:
            cldice = torch.zeros((), device=logits.device)

        total = self.bce_weight * bce + self.dice_weight * dice + self.cldice_weight * cldice
        components = {"bce": bce.item(), "dice": dice.item(), "cldice": cldice.item()}
        return total, components


def build_loss(preset: str, cldice_iterations: int = 10) -> BCEDiceClDiceLoss:
    if preset not in LOSS_PRESETS:
        raise ValueError(f"Unknown loss preset {preset!r}, choose from {list(LOSS_PRESETS)}")
    return BCEDiceClDiceLoss(**LOSS_PRESETS[preset], cldice_iterations=cldice_iterations)
