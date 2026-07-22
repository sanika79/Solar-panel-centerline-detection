"""Run inference + vectorization over a split, writing predicted SVGs in
the same schema as the ground-truth labels.

Usage:
    python scripts/predict.py --checkpoint outputs/checkpoints/main/best.pt \
        --data-dir "<CenterLine_Dataset path>" --manifest outputs/manifest/tiles_split.csv \
        --split val --out outputs/predictions/val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.svg_io import write_svg_segments
from model.infer import load_checkpoint, predict_tile
from postprocess.vectorize import vectorize_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-component-px", type=int, default=15)
    parser.add_argument("--min-branch-len", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="only predict the first N tiles (for quick checks)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, device=device)

    manifest = pd.read_csv(args.manifest)
    rows = manifest[manifest["split"] == args.split]
    if args.limit:
        rows = rows.iloc[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    for i, (_, row) in enumerate(rows.iterrows(), 1):
        prob, _ = predict_tile(model, data_dir / row["image_path"], device=device)
        segments = vectorize_mask(
            prob,
            threshold=args.threshold,
            min_component_px=args.min_component_px,
            min_branch_len=args.min_branch_len,
        )
        write_svg_segments(out_dir / f"{row['tile_id']}.svg", segments)
        if i % 20 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] {row['tile_id']}: {len(segments)} segments")

    print(f"\nWrote {len(rows)} predicted SVGs to {out_dir}")


if __name__ == "__main__":
    main()
