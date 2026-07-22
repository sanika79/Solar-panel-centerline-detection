"""End-to-end preprocessing entrypoint: build manifest -> spatial split ->
cache rasterized masks (thickness ablation) -> qualitative previews.

Usage:
    python scripts/preprocess.py --data-dir "<path to CenterLine_Dataset>" --out outputs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.manifest import build_manifest
from data.rasterize import rasterize_svg
from data.split import spatial_split, split_report
from data.tile_io import load_tile

THICKNESSES = (1, 3, 5)


def cache_masks(manifest: pd.DataFrame, data_dir: Path, masks_dir: Path) -> None:
    for thickness in THICKNESSES:
        out_dir = masks_dir / f"thickness_{thickness}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for _, row in manifest.iterrows():
            mask = rasterize_svg(data_dir / row["label_path"], thickness=thickness)
            plt.imsave(out_dir / f"{row['tile_id']}.png", mask * 255, cmap="gray")


def make_previews(manifest: pd.DataFrame, data_dir: Path, preview_dir: Path) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)

    picks = {
        "empty": manifest[manifest.is_empty].iloc[0],
        "normal_full_height": manifest[
            (~manifest.is_empty) & (~manifest.has_drive_pier) & (~manifest.is_boundary)
        ].iloc[0],
        "drive_pier": manifest[manifest.has_drive_pier].iloc[0],
        "boundary": manifest[manifest.is_boundary].iloc[0],
    }

    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
    for row_i, (label, row) in enumerate(picks.items()):
        img, valid_mask = load_tile(data_dir / row["image_path"])
        mask = rasterize_svg(data_dir / row["label_path"], thickness=3)

        overlay = img.copy()
        overlay[mask > 0] = [255, 0, 0]

        axes[row_i, 0].imshow(img)
        axes[row_i, 0].set_title(f"{label}\n{row['tile_id']}", fontsize=9)
        axes[row_i, 1].imshow(mask, cmap="gray")
        axes[row_i, 1].set_title("GT mask (thickness=3)", fontsize=9)
        axes[row_i, 2].imshow(overlay)
        axes[row_i, 2].imshow(np.where(valid_mask, np.nan, 1), cmap="autumn", alpha=0.4)
        axes[row_i, 2].set_title("overlay (padded region tinted)", fontsize=9)
        for ax in axes[row_i]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(preview_dir / "preprocessing_preview.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--buffer-cols", type=int, default=1)
    parser.add_argument("--skip-mask-cache", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    manifest_dir = out_dir / "manifest"
    masks_dir = out_dir / "masks"
    preview_dir = out_dir / "preview"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building tile manifest...")
    manifest = build_manifest(data_dir)

    print("[2/4] Computing spatial (col_idx-block) train/val/test split...")
    split_df = spatial_split(
        manifest,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        buffer_cols=args.buffer_cols,
    )
    split_df.to_csv(manifest_dir / "tiles_split.csv", index=False)
    print(split_report(split_df))

    if not args.skip_mask_cache:
        print(f"[3/4] Rasterizing masks at thicknesses {THICKNESSES}...")
        cache_masks(manifest, data_dir, masks_dir)
    else:
        print("[3/4] Skipping mask cache (--skip-mask-cache)")

    print("[4/4] Generating qualitative preprocessing preview...")
    make_previews(manifest, data_dir, preview_dir)

    print(f"\nDone. Manifest+split: {manifest_dir}, masks: {masks_dir}, preview: {preview_dir}")


if __name__ == "__main__":
    main()
