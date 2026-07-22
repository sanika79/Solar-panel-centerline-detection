"""Evaluate predicted SVGs against ground truth: buffered pixel P/R/F1 +
line-level matching, with breakdowns by is_empty/has_drive_pier/is_boundary.

Usage:
    python scripts/evaluate.py --data-dir "<CenterLine_Dataset path>" \
        --manifest outputs/manifest/tiles_split.csv --pred-dir outputs/predictions/val \
        --split val --out outputs/metrics_val.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.rasterize import rasterize_segments
from data.svg_io import parse_svg_segments
from data.tile_io import load_tile
from eval.metrics import buffered_prf1, match_segments


def evaluate_split(
    manifest: pd.DataFrame,
    data_dir: str | Path,
    pred_dir: str | Path,
    split: str,
    tolerance_px: int = 2,
    angle_tol_deg: float = 5.0,
    offset_tol_px: float = 5.0,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    pred_dir = Path(pred_dir)
    rows = manifest[manifest["split"] == split]

    records = []
    for _, row in rows.iterrows():
        tile_id = row["tile_id"]
        gt_segments = parse_svg_segments(data_dir / row["label_path"])
        pred_path = pred_dir / f"{tile_id}.svg"
        pred_segments = parse_svg_segments(pred_path) if pred_path.exists() else []

        _, valid_mask = load_tile(data_dir / row["image_path"])
        gt_mask = rasterize_segments(gt_segments, thickness=1).astype(bool) & valid_mask
        pred_mask = rasterize_segments(pred_segments, thickness=1).astype(bool) & valid_mask

        prf1 = buffered_prf1(pred_mask, gt_mask, tolerance_px=tolerance_px)
        m = match_segments(gt_segments, pred_segments, angle_tol_deg=angle_tol_deg, offset_tol_px=offset_tol_px)

        records.append(
            {
                "tile_id": tile_id,
                "is_empty": bool(row["is_empty"]),
                "has_drive_pier": bool(row["has_drive_pier"]),
                "is_boundary": bool(row["is_boundary"]),
                "pixel_precision": prf1["precision"],
                "pixel_recall": prf1["recall"],
                "pixel_f1": prf1["f1"],
                "n_gt_segments": m["n_gt"],
                "n_pred_segments": m["n_pred"],
                "n_matched": m["n_matched"],
                "line_match_rate": m["match_rate"],
                "line_precision": m["precision"],
                "mean_angle_error": m["mean_angle_error"],
                "mean_offset_error": m["mean_offset_error"],
            }
        )

    return pd.DataFrame(records)


def summarize(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    agg = {
        "pixel_precision": "mean",
        "pixel_recall": "mean",
        "pixel_f1": "mean",
        "line_match_rate": "mean",
        "line_precision": "mean",
        "mean_angle_error": "mean",
        "mean_offset_error": "mean",
        "n_gt_segments": "sum",
        "n_pred_segments": "sum",
        "n_matched": "sum",
    }
    if group_col:
        out = df.groupby(group_col).agg(agg)
        out["n_tiles"] = df.groupby(group_col).size()
        return out
    out = df.agg(agg)
    out["n_tiles"] = len(df)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--tolerance-px", type=int, default=2)
    parser.add_argument("--angle-tol-deg", type=float, default=5.0)
    parser.add_argument("--offset-tol-px", type=float, default=5.0)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    df = evaluate_split(
        manifest,
        args.data_dir,
        args.pred_dir,
        args.split,
        tolerance_px=args.tolerance_px,
        angle_tol_deg=args.angle_tol_deg,
        offset_tol_px=args.offset_tol_px,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote per-tile metrics to {args.out}\n")

    pd.set_option("display.width", 160)
    print("=== Overall ===")
    print(summarize(df))
    print("\n=== By is_empty (note: empty-vs-empty tiles trivially score angle/offset error = 0) ===")
    print(summarize(df, "is_empty"))
    print("\n=== By has_drive_pier ===")
    print(summarize(df, "has_drive_pier"))
    print("\n=== By is_boundary ===")
    print(summarize(df, "is_boundary"))


if __name__ == "__main__":
    main()
