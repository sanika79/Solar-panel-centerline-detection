"""Vectorize a predicted probability mask into line segments.

threshold -> skeletonize -> pixel-adjacency graph -> prune short spurs ->
per connected component, take the longest path between its two farthest
skeleton endpoints and fit a straight line through it (total-least-squares
via PCA/SVD, not ordinary y=f(x) regression, which degenerates for the
near-vertical lines in this dataset).

Deliberately does NOT bridge separate connected components together, even
if colinear and close -- that's exactly the drive-pier gap convention the
ground-truth labels use (see manifest.has_drive_pier), so bridging would
"fix" something the labels intentionally leave broken.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

Point = tuple[float, float]
Segment = tuple[Point, Point]


def _build_skeleton_graph(skeleton: np.ndarray) -> nx.Graph:

    ## Use BFS to find connected components and build a graph of pixel adjacency
    ys, xs = np.nonzero(skeleton)
    coords = set(zip(ys.tolist(), xs.tolist()))
    g = nx.Graph()
    g.add_nodes_from(coords)
    for y, x in coords:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                n = (y + dy, x + dx)
                if n in coords:
                    g.add_edge((y, x), n)
    return g


def _prune_spurs(g: nx.Graph, min_branch_len: int = 10) -> nx.Graph:
    """Iteratively remove short dead-end branches (leaf chains shorter
    than min_branch_len px hanging off a real branch point) -- raw
    skeletonize() output is littered with tiny spurs from boundary noise
    that would otherwise get fit as spurious extra segments. Isolated
    short components (no attached branch point) are left for the caller's
    min_component_px filter instead, not removed here.
    """
    g = g.copy()
    changed = True
    while changed:
        changed = False
        leaves = [n for n in g.nodes if g.degree(n) == 1]
        for leaf in leaves:
            if leaf not in g:
                continue
            path = [leaf]
            cur, prev = leaf, None
            while True:
                neighbors = [n for n in g.neighbors(cur) if n != prev]
                if len(neighbors) != 1:
                    break
                prev, cur = cur, neighbors[0]
                path.append(cur)
                if g.degree(cur) != 2:
                    break
            if len(path) < min_branch_len and g.degree(path[-1]) > 2:
                g.remove_nodes_from(path[:-1])
                changed = True
    return g


def _longest_path_in_component(g: nx.Graph, nodes: set) -> list[tuple[int, int]]:
    sub = g.subgraph(nodes)
    endpoints = [n for n in sub.nodes if sub.degree(n) <= 1] or [next(iter(sub.nodes))]
    best_path: list[tuple[int, int]] = []
    for a in endpoints:
        lengths = nx.single_source_shortest_path_length(sub, a)
        far_node = max(lengths, key=lengths.get)
        path = nx.shortest_path(sub, a, far_node)
        if len(path) > len(best_path):
            best_path = path
    return best_path


def _fit_line(points_yx: list[tuple[int, int]]) -> Segment:
    """Total-least-squares line fit via SVD, robust for near-vertical
    lines where ordinary y=f(x) regression degenerates."""
    pts = np.array([(x, y) for y, x in points_yx], dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    t = centered @ direction
    p1 = centroid + direction * t.min()
    p2 = centroid + direction * t.max()
    return (tuple(p1), tuple(p2))


def vectorize_mask(
    prob_mask: np.ndarray,
    threshold: float = 0.5,
    min_component_px: int = 15,
    min_branch_len: int = 10,
) -> list[Segment]:
    """prob_mask: (H, W) float probabilities (or a hard 0/1 mask) in
    [0, 1]. Returns a list of (start, end) segments in (x, y) coords."""
    ## Binarize 
    binary = prob_mask >= threshold

    ## Skeletonize
    skeleton = skeletonize(binary)

    ## Build pixel-adjacency graph and prune short spurs
    g = _build_skeleton_graph(skeleton)
    g = _prune_spurs(g, min_branch_len=min_branch_len)

    ## For each connected component, take the longest path between its
    ## two farthest skeleton endpoints and fit a straight line through it.

    segments: list[Segment] = []
    for component in nx.connected_components(g):
        if len(component) < min_component_px:
            continue
        path = _longest_path_in_component(g, component)
        if len(path) < 2:
            continue
        segments.append(_fit_line(path))
    return segments
