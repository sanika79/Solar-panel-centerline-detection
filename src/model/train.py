"""Train the U-Net centerline segmenter.

Usage:
    python -m src.model.train --data-dir "<CenterLine_Dataset path>" \
        --manifest outputs/manifest/tiles_split.csv --out outputs/checkpoints/main \
        --loss bce_dice_cldice --encoder resnet34 --epochs 30
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import SolarCenterlineDataset
from model.losses import build_loss
from model.unet import build_model


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_bce, total_dice, total_cldice, n_batches = 0.0, 0.0, 0.0, 0.0, 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)

            logits = model(image)
            loss, components = loss_fn(logits, mask, valid)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_bce += components["bce"]
            total_dice += components["dice"]
            total_cldice += components["cldice"]
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "bce": total_bce / n_batches,
        "dice": total_dice / n_batches,
        "cldice": total_cldice / n_batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True, help="tiles_split.csv path")
    parser.add_argument("--out", required=True, help="checkpoint output dir")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--loss", default="bce_dice_cldice", choices=["bce", "bce_dice", "bce_dice_cldice"])
    parser.add_argument("--thickness", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pier-oversample-weight", type=float, default=1.0,
        help="relative sampling weight for has_drive_pier==True train tiles vs. others "
             "(1.0 = no oversampling, uniform random sampling; >1 oversamples drive-pier tiles)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    manifest = pd.read_csv(args.manifest)
    train_ds = SolarCenterlineDataset(manifest, args.data_dir, split="train", thickness=args.thickness)
    val_ds = SolarCenterlineDataset(manifest, args.data_dir, split="val", thickness=args.thickness, seed=args.seed)

    if args.pier_oversample_weight != 1.0:
        weights = train_ds.rows["has_drive_pier"].map(
            {True: args.pier_oversample_weight, False: 1.0}
        ).to_numpy()
        pier_share = weights[train_ds.rows["has_drive_pier"]].sum() / weights.sum()
        print(
            f"Drive-pier oversampling: weight={args.pier_oversample_weight}x -> "
            f"drive-pier tiles are ~{pier_share*100:.1f}% of each epoch's draws "
            f"(natural rate: {train_ds.rows['has_drive_pier'].mean()*100:.1f}%)"
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(train_ds), replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(encoder_name=args.encoder).to(device)
    loss_fn = build_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_bce", "train_dice", "train_cldice",
             "val_loss", "val_bce", "val_dice", "val_cldice", "lr", "seconds"]
        )

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        scheduler.step(val_metrics["loss"])
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{epoch}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"(bce={val_metrics['bce']:.4f} dice={val_metrics['dice']:.4f} cldice={val_metrics['cldice']:.4f}) "
            f"lr={lr:.2e} {dt:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_metrics["loss"], train_metrics["bce"], train_metrics["dice"], train_metrics["cldice"],
                 val_metrics["loss"], val_metrics["bce"], val_metrics["dice"], val_metrics["cldice"], lr, dt]
            )

        torch.save(
            {"model_state": model.state_dict(), "encoder": args.encoder, "epoch": epoch, "args": vars(args)},
            out_dir / "last.pt",
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {"model_state": model.state_dict(), "encoder": args.encoder, "epoch": epoch, "args": vars(args)},
                out_dir / "best.pt",
            )
            print(f"  -> new best (val_loss={best_val_loss:.4f}), saved {out_dir / 'best.pt'}")

    print(f"\nDone. Best val_loss={best_val_loss:.4f}. Log: {log_path}")


if __name__ == "__main__":
    main()
