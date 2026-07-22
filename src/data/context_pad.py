"""Optional: pad a tile with real neighboring-tile pixel strips instead of
mirror/edge padding, so rows truncated at a tile boundary get a bit more
context. The tile grid is confirmed contiguous (adjacent tiles' geographic
bounding boxes share an exact edge, verified earlier against
tiles_index.csv), so real neighbor pixels are just a disk read away rather
than fabricated.

Corners (the four pad x pad squares diagonally outside the tile) are left
edge-replicated rather than fetched from diagonal neighbors -- rows here
run near-vertically, so the top/bottom strips (where a row gets truncated)
matter far more than the corners, and skipping diagonal lookups keeps this
optional feature simple.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .tile_io import TILE_SIZE, load_tile


def load_tile_with_context(
    manifest: pd.DataFrame,
    data_dir: str | Path,
    row_idx: int,
    col_idx: int,
    pad: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (image, valid_mask), both (TILE_SIZE + 2*pad) square.

    The center TILE_SIZE region is the tile itself (edge-padded to full
    size first if it's a boundary tile). The `pad`-wide top/bottom/left/
    right borders are filled from the real pixels of the neighboring
    tiles where present in the manifest; corners and any missing-neighbor
    side fall back to edge-replication. valid_mask marks which pixels are
    real tile content (own or a real neighbor) vs synthetic padding.
    """
    data_dir = Path(data_dir)
    lookup = {
        (r, c): p
        for r, c, p in zip(manifest["row_idx"], manifest["col_idx"], manifest["image_path"])
    }
    if (row_idx, col_idx) not in lookup:
        raise KeyError(f"(row_idx={row_idx}, col_idx={col_idx}) not in manifest")

    center_img, center_valid = load_tile(data_dir / lookup[(row_idx, col_idx)])

    size = TILE_SIZE + 2 * pad
    canvas = np.pad(center_img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    valid = np.zeros((size, size), dtype=bool)
    valid[pad : pad + TILE_SIZE, pad : pad + TILE_SIZE] = center_valid

    neighbor_offsets = {
        "top": (-TILE_SIZE, 0),
        "bottom": (TILE_SIZE, 0),
        "left": (0, -TILE_SIZE),
        "right": (0, TILE_SIZE),
    }
    for side, (dr, dc) in neighbor_offsets.items():
        n_key = (row_idx + dr, col_idx + dc)
        if n_key not in lookup:
            continue
        n_img, n_valid = load_tile(data_dir / lookup[n_key])
        if side == "top":
            canvas[:pad, pad : pad + TILE_SIZE] = n_img[-pad:, :]
            valid[:pad, pad : pad + TILE_SIZE] = n_valid[-pad:, :]
        elif side == "bottom":
            canvas[pad + TILE_SIZE :, pad : pad + TILE_SIZE] = n_img[:pad, :]
            valid[pad + TILE_SIZE :, pad : pad + TILE_SIZE] = n_valid[:pad, :]
        elif side == "left":
            canvas[pad : pad + TILE_SIZE, :pad] = n_img[:, -pad:]
            valid[pad : pad + TILE_SIZE, :pad] = n_valid[:, -pad:]
        elif side == "right":
            canvas[pad : pad + TILE_SIZE, pad + TILE_SIZE :] = n_img[:, :pad]
            valid[pad : pad + TILE_SIZE, pad + TILE_SIZE :] = n_valid[:, :pad]

    return canvas, valid
