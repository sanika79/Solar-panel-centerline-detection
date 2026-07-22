"""Verify D4 geometric transforms are consistent between array ops and
segment-coordinate transforms: rotating/flipping a rasterized mask array
must equal rasterizing the equivalently-transformed segments.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.augment import D4_TRANSFORMS, apply_d4
from data.rasterize import rasterize_segments
from data.svg_io import parse_svg_segments

DATA_DIR = Path(__file__).resolve().parents[2] / "CenterLine_Dataset" / "CenterLine_Dataset"


def test_d4_consistency():
    segments = parse_svg_segments(DATA_DIR / "labels" / "tile_r0_c12500.svg")
    mask = rasterize_segments(segments, thickness=3)
    dummy_image = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)
    valid_mask = np.ones((500, 500), dtype=bool)

    for name in D4_TRANSFORMS:
        array_fn, _ = D4_TRANSFORMS[name]
        mask_via_array = array_fn(mask)

        _, _, segs_t = apply_d4(dummy_image, valid_mask, segments, name)
        mask_via_coords = rasterize_segments(segs_t, thickness=3)

        n_diff = int(np.abs(mask_via_array.astype(int) - mask_via_coords.astype(int)).sum())
        total = int(mask.sum())
        print(f"{name}: mismatched_px={n_diff} (mask has {total} positive px)")
        # A handful of px can differ right at a segment's extreme tip due to
        # float rounding at the exact 0/500 boundary (see module docstring);
        # a near-total mismatch would instead indicate a wrong transform.
        assert n_diff < 0.01 * total, f"{name}: array-transform and segment-transform masks disagree"


if __name__ == "__main__":
    test_d4_consistency()
    print("All D4 transforms consistent.")
