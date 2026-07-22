"""Evaluation metrics: buffered pixel-level P/R/F1, and line-level
matching by angle + perpendicular offset + extent overlap.

Raw pixel IoU is meaningless for a 1-3px-wide line -- a single-pixel
sub-pixel-accurate offset would score as a near-total miss -- so pixel
metrics use a buffered/relaxed definition instead. Line-level matching is
the more physically meaningful metric ("is this the same row, at the
right position and tilt"), and is what actually reflects the drive-pier
gap convention: a predicted fragment only "matches" a GT segment if it
overlaps that segment's own extent along the line, not just any nearby
parallel segment, so incorrectly bridging (or spuriously breaking) a row
is penalized as a mismatch rather than silently accepted.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment

Point = tuple[float, float]
Segment = tuple[Point, Point]


# ---------- pixel-level ----------


def buffered_prf1(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_px: int = 2) -> dict:
    """A predicted pixel counts as correct if it's within `tolerance_px`
    of *some* GT pixel (symmetrically for recall)."""
    struct = np.ones((2 * tolerance_px + 1, 2 * tolerance_px + 1), dtype=bool)
    gt_dilated = binary_dilation(gt_mask, structure=struct)
    pred_dilated = binary_dilation(pred_mask, structure=struct)

    pred_px = int(pred_mask.sum())
    gt_px = int(gt_mask.sum())

    if pred_px == 0 and gt_px == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if pred_px == 0 or gt_px == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = float((pred_mask & gt_dilated).sum()) / pred_px
    recall = float((gt_mask & pred_dilated).sum()) / gt_px
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------- line-level ----------


def _segment_angle_deg(seg: Segment) -> float:
    (x1, y1), (x2, y2) = seg
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)


def _angle_diff(a1: float, a2: float) -> float:
    d = abs(a1 - a2) % 180
    return min(d, 180 - d)


def _line_params(seg: Segment) -> tuple[np.ndarray, np.ndarray]:
    """Return (point_on_line, unit_direction)."""
    (x1, y1), (x2, y2) = seg
    p = np.array([x1, y1], dtype=np.float64)
    d = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    norm = np.linalg.norm(d)
    d = d / norm if norm > 1e-9 else np.array([0.0, 1.0])
    return p, d


def _point_line_distance(point: np.ndarray, line_point: np.ndarray, line_dir: np.ndarray) -> float:
    v = point - line_point
    perp = v - np.dot(v, line_dir) * line_dir
    return float(np.linalg.norm(perp))


def _midpoint(seg: Segment) -> np.ndarray:
    return np.array([(seg[0][0] + seg[1][0]) / 2, (seg[0][1] + seg[1][1]) / 2])


def _perp_offset(seg_a: Segment, seg_b: Segment) -> float:
    """Average of both segments' midpoint-to-other-line perpendicular distance."""
    pa, da = _line_params(seg_a)
    pb, db = _line_params(seg_b)
    d1 = _point_line_distance(_midpoint(seg_a), pb, db)
    d2 = _point_line_distance(_midpoint(seg_b), pa, da)
    return (d1 + d2) / 2


def _extent_overlap_iou(seg_a: Segment, seg_b: Segment) -> float:
    """1D interval IoU: project both segments onto seg_a's own direction
    and measure span overlap -- without this, a short predicted fragment
    far along the same infinite line as a GT segment would still count as
    a match on angle+offset alone."""
    p, d = _line_params(seg_a)

    def project(seg: Segment) -> tuple[float, float]:
        t1 = float(np.dot(np.array(seg[0]) - p, d))
        t2 = float(np.dot(np.array(seg[1]) - p, d))
        return min(t1, t2), max(t1, t2)

    a_lo, a_hi = project(seg_a)
    b_lo, b_hi = project(seg_b)
    inter = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
    union = (a_hi - a_lo) + (b_hi - b_lo) - inter
    return inter / union if union > 0 else 0.0


def match_segments(
    gt_segments: list[Segment],
    pred_segments: list[Segment],
    angle_tol_deg: float = 5.0,
    offset_tol_px: float = 5.0,
) -> dict:
    """One-to-one matching (Hungarian assignment) restricted to candidate
    pairs passing angle/offset/extent-overlap gates."""
    n_gt, n_pred = len(gt_segments), len(pred_segments)
    if n_gt == 0 and n_pred == 0:
        return {
            "match_rate": 1.0, "precision": 1.0, "n_gt": 0, "n_pred": 0, "n_matched": 0,
            "mean_angle_error": 0.0, "mean_offset_error": 0.0,
        }
    if n_gt == 0 or n_pred == 0:
        return {
            "match_rate": 0.0, "precision": 0.0, "n_gt": n_gt, "n_pred": n_pred, "n_matched": 0,
            "mean_angle_error": None, "mean_offset_error": None,
        }

    cost = np.full((n_gt, n_pred), 1e6)
    valid = np.zeros((n_gt, n_pred), dtype=bool)
    angle_errs = np.zeros((n_gt, n_pred))
    offset_errs = np.zeros((n_gt, n_pred))

    for i, g in enumerate(gt_segments):
        ag = _segment_angle_deg(g)
        for j, p in enumerate(pred_segments):
            ad = _angle_diff(ag, _segment_angle_deg(p))
            od = _perp_offset(g, p)
            angle_errs[i, j] = ad
            offset_errs[i, j] = od
            if ad <= angle_tol_deg and od <= offset_tol_px and _extent_overlap_iou(g, p) > 0:
                valid[i, j] = True
                cost[i, j] = ad / angle_tol_deg + od / offset_tol_px

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_pairs = [(i, j) for i, j in zip(row_ind, col_ind) if valid[i, j]]

    n_matched = len(matched_pairs)
    mean_angle = float(np.mean([angle_errs[i, j] for i, j in matched_pairs])) if n_matched else None
    mean_offset = float(np.mean([offset_errs[i, j] for i, j in matched_pairs])) if n_matched else None

    return {
        "match_rate": n_matched / n_gt,
        "precision": n_matched / n_pred,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_matched": n_matched,
        "mean_angle_error": mean_angle,
        "mean_offset_error": mean_offset,
    }
