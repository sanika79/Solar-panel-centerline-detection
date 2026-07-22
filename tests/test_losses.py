"""Sanity-check the loss components, especially that clDice actually
discriminates connectivity, not just area -- construct two predictions
against the same target line with the identical number of pixels removed
(so plain Dice loss is ~equal for both), one removed as a middle gap
(disconnects the line into two pieces) and one removed as an end
truncation (stays a single connected piece, just shorter). clDice should
penalize the disconnected one more.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.losses import masked_bce, masked_dice, soft_cldice


def _to_logits(binary: torch.Tensor, scale: float = 20.0) -> torch.Tensor:
    return (binary * 2 - 1) * scale


def test_perfect_prediction_near_zero_loss():
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, :, 15] = 1.0  # a vertical line
    logits = _to_logits(target)
    pred = torch.sigmoid(logits)
    valid = torch.ones_like(target)

    bce = masked_bce(logits, target, valid)
    dice = masked_dice(pred, target, valid)
    cldice = soft_cldice(pred, target)

    print(f"perfect: bce={bce.item():.4f} dice={dice.item():.4f} cldice={cldice.item():.4f}")
    assert bce.item() < 1e-3
    assert dice.item() < 1e-3
    assert cldice.item() < 1e-2


def test_all_invalid_mask_no_nan():
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, :, 8] = 1.0
    logits = torch.randn(1, 1, 16, 16)
    valid = torch.zeros_like(target)  # nothing valid

    bce = masked_bce(logits, target, valid)
    dice = masked_dice(torch.sigmoid(logits), target, valid)
    assert torch.isfinite(bce), bce
    assert torch.isfinite(dice), dice
    print(f"all-invalid mask: bce={bce.item():.4f} dice={dice.item():.4f} (finite, no NaN) OK")


def test_cldice_penalizes_broken_blob_more_than_uniform_erosion():
    """Target has real thickness (3px, matching our actual thickness=3
    rasterization) so skeletonization can distinguish the two cases -- a
    thin 1px target line has no thickness for skeletonize to exploit,
    which is why a first version of this test (comparing gaps in an
    already-thin line) found no difference between break location: with
    nothing to erode, skel(pred) == pred regardless of where gaps are.

    (a) eroded: uniformly thinned to 1px (the target's own skeleton),
        full length, single connected piece.
    (b) excised: full 3px thickness kept, but a chunk removed from the
        middle -> two disconnected blobs.
    Both are subsets of target with the *same pixel count*, which forces
    identical Dice (intersection == pred's own area either way) -- so any
    difference in the loss can only come from clDice.
    """
    size = 40
    target = torch.zeros(1, 1, size, size)
    target[:, :, 5:35, 19:22] = 1.0  # 3px-wide, 30px-long vertical block

    eroded = torch.zeros(1, 1, size, size)
    eroded[:, :, 5:35, 20] = 1.0  # target's own skeleton: col 20, all 30 rows

    excised = torch.zeros(1, 1, size, size)
    excised[:, :, 5:10, 19:22] = 1.0  # top 5 rows, full width
    excised[:, :, 30:35, 19:22] = 1.0  # bottom 5 rows, full width

    assert eroded.sum() == excised.sum() == 30.0

    valid = torch.ones_like(target)
    dice_eroded = masked_dice(eroded, target, valid)
    dice_excised = masked_dice(excised, target, valid)
    print(f"dice: eroded={dice_eroded.item():.4f} excised={dice_excised.item():.4f}")
    assert abs(dice_eroded.item() - dice_excised.item()) < 1e-6, "Dice should be identical (same area kept)"

    cldice_eroded = soft_cldice(eroded, target)
    cldice_excised = soft_cldice(excised, target)
    print(f"cldice: eroded={cldice_eroded.item():.4f} excised={cldice_excised.item():.4f}")
    assert cldice_eroded.item() < cldice_excised.item(), (
        "clDice should penalize the disconnected (excised) blob more than the "
        "uniformly-thinned-but-connected (eroded) one, even though Dice treats them identically"
    )


if __name__ == "__main__":
    test_perfect_prediction_near_zero_loss()
    test_all_invalid_mask_no_nan()
    test_cldice_penalizes_broken_blob_more_than_uniform_erosion()
    print("\nAll loss tests passed.")
