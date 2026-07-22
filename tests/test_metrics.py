"""Validate eval metrics: perfect-match sanity checks, gating behavior
(angle/offset tolerance actually rejects wrong matches), and -- most
important -- that fragmentation shows up as a precision hit, since that's
exactly the failure mode seen in the real model's predictions.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.rasterize import rasterize_segments
from data.svg_io import parse_svg_segments
from eval.metrics import buffered_prf1, match_segments

DATA_DIR = Path(__file__).resolve().parents[2] / "CenterLine_Dataset" / "CenterLine_Dataset"


def test_perfect_match_pixel_and_line():
    segs = [((100.0, 0.0), (100.0, 500.0)), ((300.0, 0.0), (302.0, 500.0))]
    mask = rasterize_segments(segs, thickness=1)

    prf1 = buffered_prf1(mask, mask, tolerance_px=2)
    assert prf1["precision"] == prf1["recall"] == prf1["f1"] == 1.0

    m = match_segments(segs, segs)
    assert m["match_rate"] == 1.0 and m["precision"] == 1.0 and m["n_matched"] == 2
    assert m["mean_angle_error"] < 1e-6
    assert m["mean_offset_error"] < 1e-6
    print("perfect match: pixel F1=1.0, line match_rate=1.0 OK")


def test_both_empty_is_perfect_not_undefined():
    empty = np.zeros((500, 500), dtype=bool)
    prf1 = buffered_prf1(empty, empty)
    assert prf1["f1"] == 1.0
    m = match_segments([], [])
    assert m["match_rate"] == 1.0 and m["n_gt"] == 0
    print("both-empty: correctly scored as perfect (not 0/0 NaN) OK")


def test_missed_detection_scores_zero():
    gt = [((100.0, 0.0), (100.0, 500.0))]
    m = match_segments(gt, [])
    assert m["match_rate"] == 0.0
    print("missed detection: match_rate=0.0 OK")


def test_angle_gate_rejects_wrong_orientation():
    gt = [((100.0, 0.0), (100.0, 500.0))]  # vertical
    pred = [((0.0, 250.0), (500.0, 252.0))]  # near-horizontal, crosses the same point
    m = match_segments(gt, pred, angle_tol_deg=5.0)
    assert m["n_matched"] == 0, "a near-horizontal line must not match a vertical GT line"
    print("angle gate: wrong-orientation line correctly rejected OK")


def test_offset_gate_rejects_wrong_row():
    gt = [((100.0, 0.0), (100.0, 500.0))]
    pred = [((150.0, 0.0), (150.0, 500.0))]  # parallel, but a different row 50px away
    m = match_segments(gt, pred, angle_tol_deg=5.0, offset_tol_px=5.0)
    assert m["n_matched"] == 0, "a parallel line 50px away is a different row, must not match"
    print("offset gate: wrong-row line correctly rejected OK")


def test_fragmentation_penalizes_precision_not_recall():
    """This is the real failure mode from the trained model: one GT
    segment, but the prediction breaks it into two fragments covering its
    full extent between them. Recall-like match_rate should stay high
    (the row was found), but precision should drop (extra fragments)."""
    gt = [((100.0, 0.0), (100.0, 500.0))]
    fragmented_pred = [((100.0, 0.0), (100.0, 240.0)), ((100.0, 260.0), (100.0, 500.0))]

    m = match_segments(gt, fragmented_pred)
    print(f"fragmented: match_rate={m['match_rate']:.2f} precision={m['precision']:.2f} n_matched={m['n_matched']}")
    assert m["match_rate"] == 1.0, "the row should still count as found"
    assert m["precision"] < 1.0, "extra fragments should cost precision"
    assert m["n_matched"] == 1, "only one fragment can one-to-one match the single GT segment"


def test_real_drive_pier_tile_self_match():
    gt = parse_svg_segments(DATA_DIR / "labels" / "tile_r11000_c1500.svg")
    m = match_segments(gt, gt)
    assert m["n_gt"] == m["n_pred"] == m["n_matched"] == 8
    assert m["match_rate"] == 1.0
    print(f"real drive-pier tile self-match: {m['n_matched']}/8 OK")


if __name__ == "__main__":
    test_perfect_match_pixel_and_line()
    test_both_empty_is_perfect_not_undefined()
    test_missed_detection_scores_zero()
    test_angle_gate_rejects_wrong_orientation()
    test_offset_gate_rejects_wrong_row()
    test_fragmentation_penalizes_precision_not_recall()
    test_real_drive_pier_tile_self_match()
    print("\nAll metrics tests passed.")
