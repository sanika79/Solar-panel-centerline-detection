# Solar Tracker Centerline Detection — Report

*(Full detailed writeup with all supporting evidence: `report/METHODOLOGY_WALKTHROUGH.md`. Curated result images: `examples/`.)*

## Data analysis

The 670 tiles form a contiguous crop grid of one solar farm, not independent samples — `row_idx`/`col_idx` are pixel-grid addresses into a single orthomosaic, and every tile's `xmin/ymin/xmax/ymax` is an evaluation of one global affine transform (`Easting = 692198.97 + (col_idx+px)×GSD`, `Northing = 3528884.71 − (row_idx+py)×GSD`), confirmed by adjacent tiles' geographic edges matching bit-for-bit. Derived GSD = 4.0107 cm/px (the spec's rounded "4cm"). Of 670 tiles: 657 are the nominal 500×500px; 23 are genuinely smaller canvases at the farm's true east/south edges (confirmed via PNG IHDR — not a different GSD, as a naive first calculation wrongly suggested); 137 are legitimate empty/background tiles. A `has_drive_pier` heuristic flagged 87 tiles where a row is deliberately split around the tracker's motor assembly — a real structural feature, not a label gap to smooth over. Sampling across the whole farm confirmed every row runs near-vertically, which set the train/val/test split axis below.

## Preprocessing

`manifest.py` enriches every tile with `is_boundary`/`is_empty`/`has_drive_pier` flags feeding everything downstream. `rasterize.py` draws each label segment independently — never bridging gaps, so drive-pier breaks survive into training targets automatically. **The split runs along `col_idx`, not `row_idx`**: since rows run vertically, a row-based split would slice nearly every physical row in half between train/test, while a column-based split cuts perpendicular to them, with a one-tile buffer at each boundary (verified sufficient: max real row drift dataset-wide is 10.87px, far under the 500px buffer). Boundary tiles are edge-padded to 500×500 with a `valid_mask` so loss/metrics ignore the synthetic region. Augmentation applies all 8 D4 transforms to segment coordinates (re-rasterizing fresh, never interpolating a mask) plus color/shadow jitter, train-only.

## Model, training, and metrics

**U-Net (ResNet34, ImageNet-pretrained) + composite BCE/Dice/clDice loss.** clDice (Shit et al., CVPR 2021) penalizes broken connectivity in thin curvilinear targets, which plain Dice doesn't — the practical, lower-risk way to bring current connectivity-aware segmentation research into a proven architecture rather than betting the budget on an unproven one (e.g. LETR, Seg-Road-style hybrids). Two evaluation metrics, since raw pixel IoU is meaningless for a 1-3px line: **buffered pixel P/R/F1** (±2px tolerance) and **line-level matching** (Hungarian assignment gated on angle ≤5°, offset ≤5px, and overlapping extent — the extent gate is what correctly turns fragmentation into a precision penalty instead of letting it pass silently). Both unit-tested against known cases first.

## Key finding and fix

Stratifying the baseline model's results by `has_drive_pier` surfaced a much larger effect than general over-segmentation: line match rate collapsed from **96% (normal rows) to 41% (drive-pier tiles)**, traced to the training `DataLoader` having no stratified sampling for a pattern that's only ~12% of train data. Adding a `WeightedRandomSampler` (5x weight on drive-pier tiles, raising their epoch share to ~40%) closed part of the gap with no cost to the majority class, **confirmed on the held-out test split**:

| (test split) | no drive pier | has drive pier |
|---|---|---|
| line match rate, baseline → oversampled | 89.6% → 88.7% | 64.4% → **72.1%** |
| line precision, baseline → oversampled | 60.4% → 58.3% | 64.1% → **72.4%** |

## Extension and limitations

Implemented row-tilt/misalignment detection (median-deviation flagging) as the brief's optional extension — validated against a synthetic +8° injection (correctly flagged) with zero false positives on real farm-wide data (max natural deviation 0.96°). Main limitations: predicted masks are still jagged/fragmented relative to GT, visible in Dice/clDice plateauing well above BCE during training; a stage-by-stage diagnostic (`preliminary_results/vectorize_steps/`) suggests fragmentation from real-imagery interruptions and from drive-pier gaps are two distinct failure modes, not yet evaluated separately; and real-neighbor context-padding (`context_pad.py`, implemented) was left opt-in rather than used in the main run given the time budget.
