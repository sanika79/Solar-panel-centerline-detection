"""Validate misalignment detection: no false positives on real, clean
farm-wide GT data (including the degenerate-short-segment trap found
during development), and correct detection on a synthetic injected
anomaly.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.orientation import detect_misaligned_rows, segment_tilt_deg
from data.svg_io import parse_svg_segments

DATA_DIR = Path(__file__).resolve().parents[2] / "CenterLine_Dataset" / "CenterLine_Dataset"


def test_perfectly_vertical_segment_has_zero_tilt():
    assert abs(segment_tilt_deg(((100.0, 0.0), (100.0, 500.0)))) < 1e-9


def test_degenerate_short_segment_does_not_trigger_false_positive():
    """The real bug found during development: a ~0.3px-long GT stub in
    tile_r4000_c13000 produces a wild, meaningless tilt estimate. Must be
    filtered out by min_length_px rather than flagged as a misalignment."""
    segs = parse_svg_segments(DATA_DIR / "labels" / "tile_r4000_c13000.svg")
    results = detect_misaligned_rows(segs, threshold_deg=2.0, min_length_px=20.0)
    assert not any(r.is_misaligned for r in results), (
        "the degenerate ~0.3px stub (or anything else) got flagged -- "
        "min_length_px filtering regressed"
    )
    assert len(results) == 5, "expected the 1 degenerate stub excluded, 5 stable segments remain"


def test_no_false_positives_across_whole_farm():
    """Sweep every non-empty tile's real GT segments; nothing should
    exceed a 2 deg threshold, since natural row-to-row variation tops out
    under 1 deg everywhere in this farm (verified separately)."""
    import pandas as pd

    manifest = pd.read_csv(
        Path(__file__).resolve().parents[1] / "outputs" / "manifest" / "tiles_manifest.csv"
    )
    n_checked = 0
    for _, row in manifest[~manifest["is_empty"]].iterrows():
        segs = parse_svg_segments(DATA_DIR / row["label_path"])
        if len(segs) < 2:
            continue
        results = detect_misaligned_rows(segs, threshold_deg=2.0, min_length_px=20.0)
        n_checked += len(results)
        assert not any(r.is_misaligned for r in results), f"false positive in {row['tile_id']}"
    assert n_checked > 2000, "sanity check that the sweep actually ran over real data"
    print(f"no false positives across {n_checked} real segments")


def test_synthetic_misalignment_is_detected():
    segs = parse_svg_segments(DATA_DIR / "labels" / "tile_r0_c12500.svg")

    def rotate(seg, deg):
        (x1, y1), (x2, y2) = seg
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        theta = math.radians(deg)

        def rot(x, y):
            dx, dy = x - mx, y - my
            return (
                mx + dx * math.cos(theta) - dy * math.sin(theta),
                my + dx * math.sin(theta) + dy * math.cos(theta),
            )

        return (rot(x1, y1), rot(x2, y2))

    injected = list(segs)
    injected[2] = rotate(segs[2], 8.0)

    results = detect_misaligned_rows(injected, threshold_deg=2.0, min_length_px=20.0)
    flagged = [i for i, r in enumerate(results) if r.is_misaligned]
    assert flagged == [2], f"expected only the injected row (index 2) flagged, got {flagged}"
    assert results[2].deviation_deg > 7.0
    print(f"synthetic +8deg injection correctly flagged: deviation={results[2].deviation_deg:.2f}deg")


if __name__ == "__main__":
    test_perfectly_vertical_segment_has_zero_tilt()
    test_degenerate_short_segment_does_not_trigger_false_positive()
    test_no_false_positives_across_whole_farm()
    test_synthetic_misalignment_is_detected()
    print("\nAll orientation/misalignment tests passed.")
