"""Spatial train/val/test split along the col_idx axis.

Solar tracker rows in this farm run near-vertically (along the row_idx
axis -- confirmed across the whole footprint, x-drift of only a few px
over the full 500px tile height everywhere sampled). Splitting along
row_idx would slice essentially every physical row in half between train
and test. Splitting along col_idx instead cuts perpendicular to the rows,
so whole physical rows fall on one side of the split; a small buffer is
still dropped around each boundary to guard against the rarer case of a
row sitting close enough to a column edge to straddle two column-tiles.
"""
from __future__ import annotations

import pandas as pd


def spatial_split(
    manifest: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    buffer_cols: int = 1,
) -> pd.DataFrame:
    """Return a copy of `manifest` with a 'split' column added:
    'train' / 'val' / 'test' / 'buffer' (buffer tiles are excluded from
    all three, dropped as a guard band around split boundaries).
    """
    cols = sorted(manifest["col_idx"].unique())
    n = len(cols)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError("val_frac + test_frac leave no columns for train")

    train_cols = cols[:n_train]
    val_cols = cols[n_train : n_train + n_val]
    test_cols = cols[n_train + n_val :]

    col_to_split: dict[int, str] = {}
    col_to_split.update({c: "train" for c in train_cols})
    col_to_split.update({c: "val" for c in val_cols})
    col_to_split.update({c: "test" for c in test_cols})

    def mark_buffer(block: list[int], head: int, tail: int) -> None:
        for c in block[:head]:
            col_to_split[c] = "buffer"
        for c in (block[-tail:] if tail else []):
            col_to_split[c] = "buffer"

    # two internal boundaries: train|val and val|test
    mark_buffer(train_cols, head=0, tail=buffer_cols)
    mark_buffer(val_cols, head=buffer_cols, tail=buffer_cols)
    mark_buffer(test_cols, head=buffer_cols, tail=0)

    out = manifest.copy()
    out["split"] = out["col_idx"].map(col_to_split)
    return out


def split_report(split_df: pd.DataFrame) -> pd.DataFrame:
    """Per-split tile counts and empty/drive-pier/boundary rates, for
    sanity-checking that the split didn't produce a degenerate split
    (e.g. all-empty val set)."""
    return split_df.groupby("split").agg(
        n_tiles=("tile_id", "count"),
        n_empty=("is_empty", "sum"),
        n_drive_pier=("has_drive_pier", "sum"),
        n_boundary=("is_boundary", "sum"),
        col_idx_min=("col_idx", "min"),
        col_idx_max=("col_idx", "max"),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--buffer-cols", type=int, default=1)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    split_df = spatial_split(
        manifest,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        buffer_cols=args.buffer_cols,
    )
    split_df.to_csv(args.out, index=False)
    print(f"Wrote {len(split_df)} rows to {args.out}")
    print(split_report(split_df))
