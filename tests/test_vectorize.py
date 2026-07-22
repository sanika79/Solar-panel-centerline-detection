"""Validate vectorize_mask against known ground-truth tiles: rasterize the
real GT segments into a mask, then check we recover the same *number* of
segments (critically: drive-pier gaps must stay broken, not get bridged)
and that recovered endpoints land close to the true ones.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.rasterize import rasterize_svg
from data.svg_io import parse_svg_segments
from postprocess.vectorize import vectorize_mask

DATA_DIR = Path(__file__).resolve().parents[2] / "CenterLine_Dataset" / "CenterLine_Dataset"


def _endpoint_recall(gt_segments, pred_segments, tol=6.0):
    """Fraction of GT endpoints that have some predicted endpoint within
    `tol` px (a loose check -- line-fit endpoints won't be pixel-exact,
    especially near tile edges where skeletonize can shave a few px)."""
    pred_points = [p for seg in pred_segments for p in seg]
    hits = 0
    total = 0
    for (x1, y1), (x2, y2) in gt_segments:
        for gx, gy in [(x1, y1), (x2, y2)]:
            total += 1
            if any(((gx - px) ** 2 + (gy - py) ** 2) ** 0.5 <= tol for px, py in pred_points):
                hits += 1
    return hits / total if total else 1.0


def test_empty_tile_recovers_zero_segments():
    mask = rasterize_svg(DATA_DIR / "labels" / "tile_r0_c11000.svg", thickness=3).astype(np.float64)
    pred = vectorize_mask(mask)
    print(f"empty tile: recovered {len(pred)} segments")
    assert len(pred) == 0


def test_normal_tile_recovers_same_count_and_close_endpoints():
    gt = parse_svg_segments(DATA_DIR / "labels" / "tile_r0_c12500.svg")
    mask = rasterize_svg(DATA_DIR / "labels" / "tile_r0_c12500.svg", thickness=3).astype(np.float64)
    pred = vectorize_mask(mask)
    print(f"tile_r0_c12500: GT={len(gt)} segments, recovered={len(pred)}")
    assert len(pred) == len(gt) == 4

    recall = _endpoint_recall(gt, pred, tol=6.0)
    print(f"  endpoint recall (within 6px): {recall:.2f}")
    assert recall >= 0.75


def test_drive_pier_tile_keeps_gap_not_bridged():
    gt = parse_svg_segments(DATA_DIR / "labels" / "tile_r11000_c1500.svg")
    mask = rasterize_svg(DATA_DIR / "labels" / "tile_r11000_c1500.svg", thickness=3).astype(np.float64)
    pred = vectorize_mask(mask)
    print(f"tile_r11000_c1500 (drive pier): GT={len(gt)} segments, recovered={len(pred)}")
    # the critical check: must stay 8 broken segments, NOT collapse to 4
    # bridged full-height rows
    assert len(pred) == len(gt) == 8, (
        f"expected the drive-pier gaps to stay broken (8 segments), got {len(pred)} "
        "-- if this is 4, the pipeline is incorrectly bridging the gaps"
    )

    recall = _endpoint_recall(gt, pred, tol=6.0)
    print(f"  endpoint recall (within 6px): {recall:.2f}")
    assert recall >= 0.7


if __name__ == "__main__":
    test_empty_tile_recovers_zero_segments()
    test_normal_tile_recovers_same_count_and_close_endpoints()
    test_drive_pier_tile_keeps_gap_not_bridged()
    print("\nAll vectorize tests passed.")
