"""Load tile images, uniformly padding boundary tiles to 500x500.

23 of the 670 tiles sit on the true edge of the surveyed farm and are
genuinely smaller than 500x500 (e.g. 484x500) -- not 500x500 canvases with
a hidden NoData strip. Edge-pad (not zero-pad) so the model doesn't see an
artificial black/empty region, and carry a valid_mask so loss/metrics can
still ignore the synthetic padding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

TILE_SIZE = 500


def load_tile(image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (image, valid_mask).

    image: (TILE_SIZE, TILE_SIZE, 3) uint8, edge-padded if the source tile
        is smaller.
    valid_mask: (TILE_SIZE, TILE_SIZE) bool, True where pixels are real
        tile content as opposed to synthetic padding.
    """
    with Image.open(image_path) as im:
        arr = np.array(im.convert("RGB"))

    h, w = arr.shape[:2]
    valid_mask = np.ones((TILE_SIZE, TILE_SIZE), dtype=bool)

    if h == TILE_SIZE and w == TILE_SIZE:
        return arr, valid_mask

    pad_bottom = TILE_SIZE - h
    pad_right = TILE_SIZE - w
    if pad_bottom < 0 or pad_right < 0:
        raise ValueError(
            f"{image_path} is larger than {TILE_SIZE}x{TILE_SIZE}: {arr.shape}"
        )

    padded = np.pad(arr, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="edge")
    valid_mask[h:, :] = False
    valid_mask[:, w:] = False
    return padded, valid_mask
