"""PyTorch Dataset wiring the preprocessing primitives together:
load tile (+valid_mask) -> [optional D4 augment, train split only] ->
rasterize mask at the chosen thickness -> ImageNet-normalize -> tensors.

Augmentation is applied to the vector segments, then the mask is
rasterized fresh from the transformed segments -- consistent with the
rest of this preprocessing pipeline (see augment.py), so the target mask
is never produced by interpolating an already-rasterized array.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import color_jitter, normalize_imagenet, random_d4
from .context_pad import load_tile_with_context
from .rasterize import rasterize_segments
from .svg_io import parse_svg_segments
from .tile_io import TILE_SIZE, load_tile


class SolarCenterlineDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        data_dir: str | Path,
        split: str,
        thickness: int = 3,
        augment: bool | None = None,
        use_context_pad: bool = False,
        context_pad_px: int = 32,
        seed: int | None = None,
    ) -> None:
        """
        manifest: the enriched, split-labeled manifest (output of
            data.split.spatial_split), with an is_empty/has_drive_pier/
            is_boundary/split column set.
        split: one of 'train' / 'val' / 'test' (rows with split=='buffer'
            are never selected by any split name, by design).
        augment: defaults to True for split=='train', False otherwise --
            val/test must stay deterministic for fair evaluation, so an
            explicit augment=True on val/test is intentionally still
            allowed (e.g. for a TTA experiment) but not the default.
        use_context_pad: if True, load each tile with a `context_pad_px`
            border of real neighboring-tile pixels (see context_pad.py)
            instead of the tile alone; segments are shifted by
            +context_pad_px so the rasterized mask lines up with the
            larger canvas.
        """
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.thickness = thickness
        self.augment = augment if augment is not None else (split == "train")
        self.use_context_pad = use_context_pad
        self.context_pad_px = context_pad_px
        self.size = TILE_SIZE + (2 * context_pad_px if use_context_pad else 0)
        self._seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        rng = random.Random(self._seed + idx) if self._seed is not None else random

        if self.use_context_pad:
            image, valid_mask = load_tile_with_context(
                self.rows, self.data_dir, row["row_idx"], row["col_idx"], pad=self.context_pad_px
            )
            pad = self.context_pad_px
            segments = [
                ((x1 + pad, y1 + pad), (x2 + pad, y2 + pad))
                for (x1, y1), (x2, y2) in parse_svg_segments(self.data_dir / row["label_path"])
            ]
        else:
            image, valid_mask = load_tile(self.data_dir / row["image_path"])
            segments = parse_svg_segments(self.data_dir / row["label_path"])

        if self.augment:
            image, valid_mask, segments, _ = random_d4(image, valid_mask, segments, size=self.size, rng=rng)
            image = color_jitter(image, rng=rng)

        mask = rasterize_segments(segments, thickness=self.thickness, size=self.size)

        image_t = torch.from_numpy(normalize_imagenet(image))
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        valid_t = torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(0)

        return {
            "image": image_t,
            "mask": mask_t,
            "valid_mask": valid_t,
            "tile_id": row["tile_id"],
            "is_empty": bool(row["is_empty"]),
            "has_drive_pier": bool(row["has_drive_pier"]),
            "is_boundary": bool(row["is_boundary"]),
        }
