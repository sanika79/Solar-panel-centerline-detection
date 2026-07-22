"""Same walkthrough as explain_vectorize_steps.py, but saves each stage as
its own separate, full-size image instead of one small multi-panel grid.

Usage:
    python scripts/explain_vectorize_steps_individual.py \
        --checkpoint outputs/checkpoints/oversample_pier/best.pt \
        --data-dir "<CenterLine_Dataset path>" --tile-id tile_r4000_c19000 \
        --out-dir preliminary_results/vectorize_steps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.svg_io import parse_svg_segments
from data.tile_io import load_tile
from model.infer import load_checkpoint, predict_tile
from postprocess.vectorize import (
    _build_skeleton_graph,
    _fit_line,
    _longest_path_in_component,
    _prune_spurs,
)


def save(fig, out_dir: Path, name: str) -> None:
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-component-px", type=int, default=15)
    parser.add_argument("--min-branch-len", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, device=device)

    img_path = data_dir / "tiles" / f"{args.tile_id}.png"
    label_path = data_dir / "labels" / f"{args.tile_id}.svg"
    orig_img, _ = load_tile(img_path)
    gt_segments = parse_svg_segments(label_path)

    prob_mask, valid_mask = predict_tile(model, img_path, device=device)
    binary = prob_mask >= args.threshold
    skeleton = skeletonize(binary)
    g_raw = _build_skeleton_graph(skeleton)
    g_pruned = _prune_spurs(g_raw, min_branch_len=args.min_branch_len)
    pruned_pixels = {(y, x) for y, x in g_raw.nodes} - {(y, x) for y, x in g_pruned.nodes}

    segments, component_paths = [], []
    n_discarded_small = 0
    for component in nx.connected_components(g_pruned):
        if len(component) < args.min_component_px:
            n_discarded_small += 1
            continue
        path = _longest_path_in_component(g_pruned, component)
        if len(path) < 2:
            continue
        component_paths.append(path)
        segments.append(_fit_line(path))

    # -- 0: input tile --
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(orig_img); ax.axis("off")
    ax.set_title(f"{args.tile_id} — input tile", fontsize=13)
    save(fig, out_dir, "0_input_tile.png")

    # -- 1: raw probability mask --
    fig, ax = plt.subplots(figsize=(8.5, 8))
    im = ax.imshow(prob_mask, cmap="viridis", vmin=0, vmax=1); ax.axis("off")
    ax.set_title(f"[0] Raw model probability mask\nmean={prob_mask.mean():.4f}, max={prob_mask.max():.3f}", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, out_dir, "1_raw_probability.png")

    # -- 2: thresholded --
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(binary, cmap="gray"); ax.axis("off")
    ax.set_title(f"[1] Thresholded >= {args.threshold}\n{int(binary.sum())} px ({100*binary.mean():.2f}% of tile)", fontsize=13)
    save(fig, out_dir, "2_thresholded.png")

    # -- 3: skeleton --
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(skeleton, cmap="gray"); ax.axis("off")
    ax.set_title(f"[2] Skeletonized\n{int(skeleton.sum())} px (from {int(binary.sum())})", fontsize=13)
    save(fig, out_dir, "3_skeleton.png")

    # -- 4: graph + pruned spurs --
    graph_vis = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    for y, x in g_pruned.nodes:
        graph_vis[y, x] = [255, 255, 255]
    for y, x in pruned_pixels:
        graph_vis[y, x] = [255, 60, 60]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(graph_vis); ax.axis("off")
    n_comp_raw = nx.number_connected_components(g_raw)
    ax.set_title(
        f"[3+4] Adjacency graph, spur-pruned\nwhite=kept ({g_pruned.number_of_nodes()}px), "
        f"red=pruned ({len(pruned_pixels)}px)\n{n_comp_raw} connected components before pruning",
        fontsize=13,
    )
    save(fig, out_dir, "4_graph_pruned.png")

    # -- 5a: connected components, colored --
    comp_vis = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    cmap = plt.get_cmap("tab10")
    n_comp_pruned = nx.number_connected_components(g_pruned)
    for i, comp in enumerate(nx.connected_components(g_pruned)):
        if len(comp) < args.min_component_px:
            color = (110, 110, 110)
        else:
            c = np.array(cmap(i % 10)[:3]) * 255
            color = tuple(c.astype(int).tolist())
        for y, x in comp:
            comp_vis[y, x] = color
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(comp_vis); ax.axis("off")
    ax.set_title(
        f"[5] Connected components (each color = one component)\n"
        f"{n_comp_pruned} total, {n_discarded_small} discarded as noise (gray, <{args.min_component_px}px)",
        fontsize=13,
    )
    save(fig, out_dir, "5a_connected_components.png")

    # -- 5b: longest path per component, on the image --
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(orig_img)
    for path in component_paths:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, color="yellow", linewidth=2)
    ax.axis("off")
    ax.set_title(f"[5] Longest path per component ({len(component_paths)} paths)", fontsize=13)
    save(fig, out_dir, "5b_longest_paths.png")

    # -- 6: final fitted segments vs GT --
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(orig_img)
    for (x1, y1), (x2, y2) in gt_segments:
        ax.plot([x1, x2], [y1, y2], color="lime", linewidth=3, alpha=0.85, label="_gt")
    for (x1, y1), (x2, y2) in segments:
        ax.plot([x1, x2], [y1, y2], color="deepskyblue", linewidth=2, alpha=0.9, label="_pred")
    ax.plot([], [], color="lime", linewidth=3, label="ground truth")
    ax.plot([], [], color="deepskyblue", linewidth=2, label="fitted prediction")
    ax.legend(loc="upper right", fontsize=11)
    ax.axis("off")
    ax.set_title(f"[6] Final fitted segments: {len(segments)} predicted vs {len(gt_segments)} GT", fontsize=13)
    save(fig, out_dir, "6_final_segments_vs_gt.png")

    print(f"\nAll 8 stage images saved to {out_dir}")


if __name__ == "__main__":
    main()
