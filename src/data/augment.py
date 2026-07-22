"""Mask-consistent augmentations and pixel normalization.

Geometric augmentations (D4 dihedral group: 90/180/270 rotations composed
with an optional mirror) are applied to the raw line-segment coordinates,
not by interpolating an already-rasterized mask -- the mask is rasterized
fresh from the transformed segments afterwards, so it stays an exact
binary target instead of picking up interpolation artifacts.

Coordinate formulas use the discrete pixel-index reflection `(size - 1) -
v`, matching exactly what np.rot90/fliplr/flipud do to the array (verified
empirically in tests/test_augment.py) -- the naive continuous-looking
`size - v` looks appealing since label coordinates can be exactly 500.00,
but it is off by one full pixel from the real array operation, which for
a thin line means a full row/column shift rather than a harmless rounding
difference. `(size - 1) - v` can push an exact boundary value (e.g. y =
500.00) to -1.0 before rounding; that only clips ~1px off a line's
extreme tip and is negligible for augmentation purposes.
"""
from __future__ import annotations

import random

import numpy as np

from .svg_io import Segment

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _identity(x: float, y: float, size: int) -> tuple[float, float]:
    return x, y


def _rot90(x: float, y: float, size: int) -> tuple[float, float]:
    return y, (size - 1) - x


def _rot180(x: float, y: float, size: int) -> tuple[float, float]:
    return (size - 1) - x, (size - 1) - y


def _rot270(x: float, y: float, size: int) -> tuple[float, float]:
    return (size - 1) - y, x


def _flip_h(x: float, y: float, size: int) -> tuple[float, float]:
    return (size - 1) - x, y


def _flip_v(x: float, y: float, size: int) -> tuple[float, float]:
    return x, (size - 1) - y


# name -> (array_fn(arr) -> arr, point_fn(x, y, size) -> (x, y))
D4_TRANSFORMS = {
    "identity": (lambda a: a, _identity),
    "rot90": (lambda a: np.rot90(a, 1), _rot90),
    "rot180": (lambda a: np.rot90(a, 2), _rot180),
    "rot270": (lambda a: np.rot90(a, 3), _rot270),
    "flip_h": (lambda a: np.fliplr(a), _flip_h),
    "flip_h_rot90": (lambda a: np.rot90(np.fliplr(a), 1), lambda x, y, s: _rot90(*_flip_h(x, y, s), s)),
    "flip_h_rot180": (lambda a: np.rot90(np.fliplr(a), 2), lambda x, y, s: _rot180(*_flip_h(x, y, s), s)),
    "flip_v": (lambda a: np.flipud(a), _flip_v),
}


def apply_d4(
    image: np.ndarray,
    valid_mask: np.ndarray,
    segments: list[Segment],
    transform_name: str,
    size: int = 500,
) -> tuple[np.ndarray, np.ndarray, list[Segment]]:
    array_fn, point_fn = D4_TRANSFORMS[transform_name]
    new_image = np.ascontiguousarray(array_fn(image))
    new_valid_mask = np.ascontiguousarray(array_fn(valid_mask))
    new_segments = [
        (point_fn(x1, y1, size), point_fn(x2, y2, size)) for (x1, y1), (x2, y2) in segments
    ]
    return new_image, new_valid_mask, new_segments


def random_d4(
    image: np.ndarray,
    valid_mask: np.ndarray,
    segments: list[Segment],
    size: int = 500,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Segment], str]:
    rng = rng or random
    name = rng.choice(list(D4_TRANSFORMS))
    image_t, valid_mask_t, segments_t = apply_d4(image, valid_mask, segments, name, size)
    return image_t, valid_mask_t, segments_t, name


def color_jitter(
    image: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
    shadow_prob: float = 0.3,
    shadow_strength: tuple[float, float] = (0.4, 0.75),
    rng: random.Random | None = None,
) -> np.ndarray:
    """Brightness/contrast jitter plus an occasional synthetic shadow band,
    to emulate the shadow/sun-angle variability seen across tiles. Image
    stays uint8 in [0, 255]; only pixel values change, not geometry, so
    segments/valid_mask are untouched by this step.
    """
    rng = rng or random
    out = image.astype(np.float32)

    gain = 1.0 + rng.uniform(-brightness, brightness)
    out *= gain

    mean = out.mean()
    contrast_gain = 1.0 + rng.uniform(-contrast, contrast)
    out = (out - mean) * contrast_gain + mean

    if rng.random() < shadow_prob:
        h, w = image.shape[:2]
        band_width = rng.uniform(0.2, 0.5) * w
        band_center = rng.uniform(0, w)
        xs = np.arange(w)
        dist = np.minimum(np.abs(xs - band_center), w - np.abs(xs - band_center))
        strength = rng.uniform(*shadow_strength)
        falloff = np.clip(1 - dist / (band_width / 2), 0, 1)
        darken = 1 - strength * falloff
        out *= darken[np.newaxis, :, np.newaxis]

    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_imagenet(image_uint8: np.ndarray) -> np.ndarray:
    """uint8 HWC [0,255] -> float32 CHW normalized with ImageNet mean/std."""
    x = image_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1)).astype(np.float32)
