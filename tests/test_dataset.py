"""Sanity-check SolarCenterlineDataset: split sizes, shapes/dtypes,
val/test determinism, train augmentation actually varying, and that a
DataLoader-collated batch moves to CUDA cleanly.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from torch.utils.data import DataLoader

from data.dataset import SolarCenterlineDataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "CenterLine_Dataset" / "CenterLine_Dataset"
MANIFEST_PATH = ROOT / "solar-centerline" / "outputs" / "manifest" / "tiles_split.csv"


def test_split_sizes_and_shapes():
    manifest = pd.read_csv(MANIFEST_PATH)
    expected = {"train": 477, "val": 64, "test": 65}

    for split, n in expected.items():
        ds = SolarCenterlineDataset(manifest, DATA_DIR, split=split, thickness=3)
        assert len(ds) == n, f"{split}: expected {n} tiles, got {len(ds)}"

        sample = ds[0]
        assert sample["image"].shape == (3, 500, 500), sample["image"].shape
        assert sample["mask"].shape == (1, 500, 500), sample["mask"].shape
        assert sample["valid_mask"].shape == (1, 500, 500), sample["valid_mask"].shape
        assert sample["image"].dtype == torch.float32
        assert set(sample["mask"].unique().tolist()) <= {0.0, 1.0}
        print(f"{split}: n={len(ds)} OK, sample tile_id={sample['tile_id']}")


def test_val_is_deterministic_train_is_augmented():
    manifest = pd.read_csv(MANIFEST_PATH)

    val_ds = SolarCenterlineDataset(manifest, DATA_DIR, split="val", thickness=3)
    a = val_ds[0]["mask"]
    b = val_ds[0]["mask"]
    assert torch.equal(a, b), "val split must be deterministic (no augmentation) by default"
    print("val determinism OK")

    train_ds = SolarCenterlineDataset(manifest, DATA_DIR, split="train", thickness=3)
    # pick a non-empty tile -- an empty tile's mask is all-zero under every
    # D4 transform, which would trivially "pass" without exercising anything
    nonempty_idx = train_ds.rows.index[~train_ds.rows["is_empty"]][0]
    masks = [train_ds[nonempty_idx]["mask"] for _ in range(6)]
    images = [train_ds[nonempty_idx]["image"] for _ in range(6)]
    masks_vary = not all(torch.equal(masks[0], m) for m in masks[1:])
    images_vary = not all(torch.equal(images[0], im) for im in images[1:])
    assert masks_vary, "train mask should vary across fetches (D4 augmentation active)"
    assert images_vary, "train image should vary across fetches (color jitter active)"
    print("train augmentation varies across fetches (mask geometry + image color) OK")


def test_dataloader_and_cuda():
    manifest = pd.read_csv(MANIFEST_PATH)
    ds = SolarCenterlineDataset(manifest, DATA_DIR, split="train", thickness=3)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    assert batch["image"].shape == (8, 3, 500, 500)
    assert batch["mask"].shape == (8, 1, 500, 500)
    print("batch shapes OK:", batch["image"].shape, batch["mask"].shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = batch["image"].to(device)
    msk = batch["mask"].to(device)
    print(f"moved batch to {device} OK, image.device={img.device}, mask.device={msk.device}")


def test_empty_tile_has_all_zero_mask():
    manifest = pd.read_csv(MANIFEST_PATH)
    ds = SolarCenterlineDataset(manifest, DATA_DIR, split="train", thickness=3, augment=False)
    empty_rows = ds.rows[ds.rows["is_empty"]]
    assert len(empty_rows) > 0
    idx = ds.rows.index[ds.rows["is_empty"]][0]
    sample = ds[idx]
    assert sample["mask"].sum().item() == 0.0
    print(f"empty tile {sample['tile_id']}: mask sum = 0 OK")


if __name__ == "__main__":
    test_split_sizes_and_shapes()
    test_val_is_deterministic_train_is_augmented()
    test_dataloader_and_cuda()
    test_empty_tile_has_all_zero_mask()
    print("\nAll dataset tests passed.")
