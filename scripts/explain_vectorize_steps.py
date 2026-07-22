"""Walk through every stage of vectorize_mask() on one real prediction and
save a visual + printed stats at each step, for explaining the pipeline.

Usage:
    python scripts/explain_vectorize_steps.py --checkpoint outputs/checkpoints/oversample_pier/best.pt \
        --data-dir "<CenterLine_Dataset path>" --tile-id tile_r4000_c19000 --out preliminary_results/vectorize_steps.png
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-component-px", type=int, default=15)
    parser.add_argument("--min-branch-len", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, device=device)

    img_path = data_dir / "tiles" / f"{args.tile_id}.png"
    label_path = data_dir / "labels" / f"{args.tile_id}.svg"
    orig_img, _ = load_tile(img_path)
    gt_segments = parse_svg_segments(label_path)

    # ---- Step 0: raw model output ----
    prob_mask, valid_mask = predict_tile(model, img_path, device=device)
    print(f"[0] raw probability mask: shape={prob_mask.shape}, "
          f"min={prob_mask.min():.3f}, max={prob_mask.max():.3f}, mean={prob_mask.mean():.4f}")

    # ---- Step 1: threshold ----
    binary = prob_mask >= args.threshold
    print(f"[1] thresholded (>= {args.threshold}): {int(binary.sum())} foreground px "
          f"({100*binary.mean():.2f}% of tile)")

    # ---- Step 2: skeletonize ----
    skeleton = skeletonize(binary)
    print(f"[2] skeletonize: {int(skeleton.sum())} skeleton px "
          f"(reduced from {int(binary.sum())} -> {int(skeleton.sum())}, "
          f"{100*skeleton.sum()/max(binary.sum(),1):.1f}% retained)")

    # ---- Step 3: build pixel-adjacency graph ----
    g_raw = _build_skeleton_graph(skeleton)
    n_components_raw = nx.number_connected_components(g_raw)
    leaves_raw = sum(1 for n in g_raw.nodes if g_raw.degree(n) == 1)
    branches_raw = sum(1 for n in g_raw.nodes if g_raw.degree(n) > 2)
    print(f"[3] graph: {g_raw.number_of_nodes()} nodes, {g_raw.number_of_edges()} edges, "
          f"{n_components_raw} connected components, {leaves_raw} leaf (degree=1) nodes, "
          f"{branches_raw} branch (degree>2) nodes")

    # ---- Step 4: prune spurs ----
    g_pruned = _prune_spurs(g_raw, min_branch_len=args.min_branch_len)
    n_components_pruned = nx.number_connected_components(g_pruned)
    removed_px = g_raw.number_of_nodes() - g_pruned.number_of_nodes()
    print(f"[4] prune spurs (min_branch_len={args.min_branch_len}): removed {removed_px} px, "
          f"{g_pruned.number_of_nodes()} nodes remain, "
          f"{n_components_pruned} connected components (was {n_components_raw})")

    # ---- Step 5: per-component longest path + line fit ----
    segments = []
    component_paths = []
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
    print(f"[5] per component: {n_components_pruned} components -> "
          f"{n_discarded_small} discarded (< {args.min_component_px}px) -> "
          f"{len(segments)} line-fit segments")

    print(f"[6] final output: {len(segments)} segments (GT has {len(gt_segments)})")
    for i, ((x1, y1), (x2, y2)) in enumerate(segments):
        print(f"     seg {i}: ({x1:.1f},{y1:.1f}) -> ({x2:.1f},{y2:.1f})")

    # ---------------- visualization ----------------
    pruned_pixels = {(y, x) for y, x in g_raw.nodes} - {(y, x) for y, x in g_pruned.nodes}

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title("Input tile", fontsize=10)

    im = axes[0, 1].imshow(prob_mask, cmap="viridis", vmin=0, vmax=1)
    axes[0, 1].set_title("[0] Raw probability mask", fontsize=10)
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].imshow(binary, cmap="gray")
    axes[0, 2].set_title(f"[1] Thresholded (>={args.threshold})\n{int(binary.sum())} px", fontsize=10)

    axes[0, 3].imshow(skeleton, cmap="gray")
    axes[0, 3].set_title(f"[2] Skeletonized\n{int(skeleton.sum())} px", fontsize=10)

    graph_vis = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    for y, x in g_pruned.nodes:
        graph_vis[y, x] = [255, 255, 255]
    for y, x in pruned_pixels:
        graph_vis[y, x] = [255, 60, 60]
    axes[1, 0].imshow(graph_vis)
    axes[1, 0].set_title(
        f"[3+4] Graph + pruned spurs\nwhite=kept, red=pruned ({len(pruned_pixels)}px)", fontsize=10
    )

    comp_vis = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    cmap = plt.get_cmap("tab10")
    for i, comp in enumerate(nx.connected_components(g_pruned)):
        if len(comp) < args.min_component_px:
            color = (120, 120, 120)
        else:
            c = np.array(cmap(i % 10)[:3]) * 255
            color = tuple(c.astype(int).tolist())
        for y, x in comp:
            comp_vis[y, x] = color
    axes[1, 1].imshow(comp_vis)
    axes[1, 1].set_title(
        f"[5] Connected components\n{n_components_pruned} total, {n_discarded_small} discarded (gray)",
        fontsize=10,
    )

    axes[1, 2].imshow(orig_img)
    for path in component_paths:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        axes[1, 2].plot(xs, ys, color="yellow", linewidth=1.5)
    axes[1, 2].set_title("[5] Longest path per component", fontsize=10)

    axes[1, 3].imshow(orig_img)
    for (x1, y1), (x2, y2) in gt_segments:
        axes[1, 3].plot([x1, x2], [y1, y2], color="lime", linewidth=2, alpha=0.8)
    for (x1, y1), (x2, y2) in segments:
        axes[1, 3].plot([x1, x2], [y1, y2], color="deepskyblue", linewidth=2, alpha=0.8)
    axes[1, 3].set_title(f"[6] Final fitted segments (blue)\nvs GT (green): {len(segments)} vs {len(gt_segments)}", fontsize=10)

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(f"vectorize_mask() stage-by-stage: {args.tile_id}", fontsize=13)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
