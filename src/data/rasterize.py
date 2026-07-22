"""Rasterize centerline line segments into binary training-target masks."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .svg_io import Segment, parse_svg_segments

TILE_SIZE = 500


def rasterize_segments(
    segments: list[Segment],
    thickness: int = 3,
    size: int = TILE_SIZE,
) -> np.ndarray:
    """Draw segments as a binary {0,1} uint8 mask, `thickness` px wide.

    Drawn at 1px (cv2.line's thickness=1 is exact and version-independent),
    then dilated with an elliptical kernel to reach the requested width --
    cv2.line's own `thickness` parameter does not map 1:1 to pixel width
    across OpenCV versions (verified: on this install, thickness=3 yields
    a 5px-wide line, not 3px), so we control width ourselves instead of
    relying on it. LINE_8, not LINE_AA: anti-aliasing blends a low integer
    color (1) down to 0 at partial-coverage edge pixels, erasing thin
    lines instead of softening them.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    for (x1, y1), (x2, y2) in segments:
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.line(mask, pt1, pt2, color=1, thickness=1, lineType=cv2.LINE_8)

    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def rasterize_svg(
    svg_path: str | Path,
    thickness: int = 3,
    size: int = TILE_SIZE,
) -> np.ndarray:
    segments = parse_svg_segments(svg_path)
    return rasterize_segments(segments, thickness=thickness, size=size)
