"""Parse and write ground-truth / predicted centerline SVG labels.

Labels in this dataset are always simple 2-point polylines of the form
``<path d="M x1 y1 L x2 y2" .../>`` inside a 500x500 viewBox, so a full SVG
parser is unnecessary -- a regex extract is exact and far cheaper.
"""
from __future__ import annotations

import re
from pathlib import Path

Point = tuple[float, float]
Segment = tuple[Point, Point]

_PATH_D_RE = re.compile(r'<path\s+d="([^"]+)"')
_MOVE_LINE_RE = re.compile(
    r"M\s*([-\d.]+)\s+([-\d.]+)\s*L\s*([-\d.]+)\s+([-\d.]+)"
)


def parse_svg_segments(svg_path: str | Path) -> list[Segment]:
    """Extract all (start, end) line segments from a centerline SVG label."""
    text = Path(svg_path).read_text(encoding="utf-8")
    segments: list[Segment] = []
    for d_match in _PATH_D_RE.finditer(text):
        d = d_match.group(1)
        ml_match = _MOVE_LINE_RE.search(d)
        if ml_match is None:
            raise ValueError(f"Unrecognized path data in {svg_path}: {d!r}")
        x1, y1, x2, y2 = (float(v) for v in ml_match.groups())
        segments.append(((x1, y1), (x2, y2)))
    return segments


def write_svg_segments(
    svg_path: str | Path,
    segments: list[Segment],
    width: int = 500,
    height: int = 500,
    stroke: str = "red",
    stroke_width: float = 2.0,
) -> None:
    """Write segments back out in the same schema as the ground-truth labels."""
    paths = "".join(
        f'<path d="M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}" '
        f'fill="none" stroke="{stroke}" stroke-width="{stroke_width}" />'
        for (x1, y1), (x2, y2) in segments
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f"{paths}</svg>"
    )
    Path(svg_path).write_text(svg, encoding="utf-8")
