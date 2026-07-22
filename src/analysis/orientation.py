"""Row orientation estimation and misalignment flagging.

Every detected/GT centerline segment already implies a tilt angle (this
dataset's rows are all near-vertical, but the method is generic). Since
every row on a single-axis tracker farm shares one physical tracking axis
and moves together through the day, any row whose tilt deviates
meaningfully from its neighbors *at the same moment* is a strong
misalignment signal -- a stuck actuator, broken drive motor, or physical
damage -- independent of whatever angle the whole farm happens to be
tracking to at that time of day. This is why a single-frame, cross-
sectional comparison (this row vs. its neighbors right now) is a
meaningful anomaly signal on its own, without needing a multi-temporal
baseline -- though a real deployment would use both (see module docstring
in scripts/detect_misalignment.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float]
Segment = tuple[Point, Point]


def segment_tilt_deg(seg: Segment) -> float:
    """Signed tilt from true vertical, in degrees (0 = perfectly
    vertical). Direction is normalized to point downward (+y) first, so
    the sign consistently reflects top-to-bottom lean direction rather
    than depending on which endpoint happened to be listed first."""
    (x1, y1), (x2, y2) = seg
    dx, dy = x2 - x1, y2 - y1
    if dy < 0:
        dx, dy = -dx, -dy
    return math.degrees(math.atan2(dx, dy))


@dataclass
class RowOrientation:
    segment: Segment
    tilt_deg: float
    deviation_deg: float
    is_misaligned: bool


def _segment_length(seg: Segment) -> float:
    (x1, y1), (x2, y2) = seg
    return math.hypot(x2 - x1, y2 - y1)


def detect_misaligned_rows(
    segments: list[Segment], threshold_deg: float = 2.0, min_length_px: float = 20.0
) -> list[RowOrientation]:
    """Flag rows whose tilt deviates from the group's median tilt by more
    than `threshold_deg`. Median (not mean) so that one or two real
    anomalies don't drag the reference angle toward themselves.

    Segments shorter than `min_length_px` are excluded before computing
    both the median and the deviations -- angle estimation is inherently
    unstable for near-zero-length segments (a tiny endpoint perturbation
    swings atan2 wildly), and in practice these tend to be degenerate
    labeling/clipping stubs rather than real independent row measurements
    (verified: the one >5 deg outlier found in this dataset's real GT
    data was a ~0.3px-long stub, not a genuine misalignment).
    """
    if not segments:
        return []
    stable_segments = [s for s in segments if _segment_length(s) >= min_length_px]
    if not stable_segments:
        return []
    tilts = [segment_tilt_deg(s) for s in stable_segments]
    segments = stable_segments
    tilts_sorted = sorted(tilts)
    n = len(tilts_sorted)
    median = (
        tilts_sorted[n // 2]
        if n % 2
        else (tilts_sorted[n // 2 - 1] + tilts_sorted[n // 2]) / 2
    )

    results = []
    for seg, tilt in zip(segments, tilts):
        deviation = abs(tilt - median)
        results.append(RowOrientation(seg, tilt, deviation, deviation > threshold_deg))
    return results
