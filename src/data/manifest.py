"""Build the enriched tile manifest from tiles_index.csv.

Adds, per tile: actual pixel dimensions, `is_boundary` (dims != 500x500),
`n_segments`, `is_empty`, and a `has_drive_pier` heuristic flag.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from .svg_io import Segment, parse_svg_segments

TILE_SIZE = 500

EDGE_TOL = 1.5  # px tolerance for "reaches the tile edge"
PARTIAL_GAP = 15.0  # px minimum shortfall to call a segment "partial"
PIER_X_TOL = 20.0  # px tolerance for pairing a row's top/bottom segments


def _segment_span(seg: Segment) -> tuple[float, float, float]:
    (x1, y1), (x2, y2) = seg
    # x_mid — the average x-position of the two endpoints. 
    # Since these rows are near-vertical, this is basically "which column is this row in."
    x_mid = (x1 + x2) / 2.0
    y_start, y_end = sorted((y1, y2))
    return x_mid, y_start, y_end


def has_drive_pier(segments: list[Segment]) -> bool:
    """A row split into a top segment and a bottom segment, at matching x,
    that don't reach the tile edge they're missing -- i.e. broken mid-tile
    rather than merely truncated by the tile boundary."""
    tops, bottoms = [], []
    for seg in segments:
        x_mid, y_start, y_end = _segment_span(seg)
        if y_start <= EDGE_TOL and y_end <= TILE_SIZE - PARTIAL_GAP:
            tops.append(x_mid)
        elif y_end >= TILE_SIZE - EDGE_TOL and y_start >= PARTIAL_GAP:
            bottoms.append(x_mid)
    return any(abs(t - b) <= PIER_X_TOL for t in tops for b in bottoms)


def build_manifest(data_dir: str | Path) -> pd.DataFrame:
    """Parse tiles_index.csv and enrich with per-tile derived flags."""
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "metadata" / "tiles_index.csv")

    widths: list[int] = []
    heights: list[int] = []
    n_segments: list[int] = []
    is_empty: list[bool] = []
    is_boundary: list[bool] = []
    drive_pier: list[bool] = []

    for _, row in df.iterrows():
        with Image.open(data_dir / row["image_path"]) as im:
            w, h = im.size
        widths.append(w)
        heights.append(h)
        is_boundary.append(w != TILE_SIZE or h != TILE_SIZE)

        segs = parse_svg_segments(data_dir / row["label_path"])
        n_segments.append(len(segs))
        is_empty.append(len(segs) == 0)
        drive_pier.append(has_drive_pier(segs))

    df["width"] = widths
    df["height"] = heights
    df["is_boundary"] = is_boundary
    df["n_segments"] = n_segments
    df["is_empty"] = is_empty
    df["has_drive_pier"] = drive_pier
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.data_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)
    print(f"Wrote {len(manifest)} rows to {args.out}")
    print(manifest[["is_boundary", "is_empty", "has_drive_pier"]].sum())
