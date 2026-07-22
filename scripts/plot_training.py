"""Plot training/val loss curves, loss-component breakdown, and LR
schedule from a train_log.csv, for reporting/explaining a training run.

Usage:
    python scripts/plot_training.py --log outputs/checkpoints/main/train_log.csv \
        --out preliminary_results/training_curves.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_log(log_path: str | Path, out_path: str | Path) -> None:
    df = pd.read_csv(log_path)
    best_epoch = df.loc[df["val_loss"].idxmin()]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], label="train_loss", color="#1f77b4", linewidth=2)
    ax.plot(df["epoch"], df["val_loss"], label="val_loss", color="#ff7f0e", linewidth=2)
    ax.scatter([best_epoch["epoch"]], [best_epoch["val_loss"]], color="#2ecc40", zorder=5, s=60)
    ax.annotate(
        f"best checkpoint\nepoch {int(best_epoch['epoch'])}, val_loss={best_epoch['val_loss']:.4f}",
        xy=(best_epoch["epoch"], best_epoch["val_loss"]),
        xytext=(best_epoch["epoch"] + 1.5, best_epoch["val_loss"] + 0.15),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#2ecc40"),
    )
    ax.set_ylabel("composite loss\n(BCE + Dice + clDice)")
    ax.set_title("Training run: U-Net (ResNet34) + BCE/Dice/clDice, thickness=3")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df["epoch"], df["val_bce"], label="val BCE", linewidth=2)
    ax.plot(df["epoch"], df["val_dice"], label="val Dice loss", linewidth=2)
    ax.plot(df["epoch"], df["val_cldice"], label="val clDice loss", linewidth=2)
    ax.set_ylabel("val loss component")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.step(df["epoch"], df["lr"], where="post", color="#9467bd", linewidth=2)
    ax.set_yscale("log")
    ax.set_ylabel("learning rate (log scale)")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(
        f"Best epoch: {int(best_epoch['epoch'])}, val_loss={best_epoch['val_loss']:.4f} "
        f"(bce={best_epoch['val_bce']:.4f}, dice={best_epoch['val_dice']:.4f}, cldice={best_epoch['val_cldice']:.4f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plot_training_log(args.log, args.out)
