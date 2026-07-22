# Solar Tracker Centerline Detection — Report

*(Full detailed writeup with all supporting evidence: `report/METHODOLOGY_WALKTHROUGH.md`. Curated result images: `examples/`.)*

## Data analysis

The 670 tiles form a contiguous crop grid of one solar farm, not independent samples — `row_idx`/`col_idx` are pixel-grid addresses into a single orthomosaic, and every tile's `xmin/ymin/xmax/ymax` is an evaluation of one global affine transform (`Easting = 692198.97 + (col_idx+px)×GSD`, `Northing = 3528884.71 − (row_idx+py)×GSD`), confirmed by adjacent tiles' geographic edges matching bit-for-bit. Derived GSD = 4.0107 cm/px (the spec's rounded "4cm"). Of 670 tiles: 657 are the nominal 500×500px; 23 are genuinely smaller canvases at the farm's true east/south edges (confirmed via PNG IHDR, not a different GSD as a naive first pass wrongly suggested); 137 are legitimate empty/background tiles. A `has_drive_pier` heuristic flagged 87 tiles where a row is deliberately split around the tracker's motor assembly — a real structural feature, not a label gap. Sampling across the whole farm confirmed every row runs near-vertically, which set the split axis below.

## Preprocessing

`manifest.py` enriches every tile with `is_boundary`/`is_empty`/`has_drive_pier` flags feeding everything downstream. `rasterize.py` draws each label segment independently — never bridging gaps, so drive-pier breaks survive into training targets automatically. **The split runs along `col_idx`, not `row_idx`**: since rows run vertically, a row-based split would slice nearly every row in half between train/test, while a column-based split cuts perpendicular to them, with a one-tile buffer at each boundary (verified sufficient: max real row drift dataset-wide is 10.87px, far under the 500px buffer). Boundary tiles are edge-padded to 500×500 with a `valid_mask` so loss/metrics ignore the synthetic region. Augmentation applies all 8 D4 transforms to segment coordinates (re-rasterized fresh, never interpolating a mask) plus color/shadow jitter, train-only.

## Model, training, and metrics

**U-Net + composite BCE/Dice/clDice loss**, encoder ablated between ResNet34 (CNN, ImageNet-pretrained) and MiT-B0 (SegFormer's transformer backbone, ~5.5M params vs. ResNet34's ~24M). clDice (Shit et al., CVPR 2021) penalizes broken connectivity in thin curvilinear targets, which plain Dice doesn't — the practical, lower-risk way to bring current connectivity-aware segmentation research into a proven architecture rather than betting the budget on an unproven one (e.g. LETR, Seg-Road-style hybrids). Two evaluation metrics, since raw pixel IoU is meaningless for a 1-3px line: **buffered pixel P/R/F1** (±2px tolerance) and **line-level matching** (Hungarian assignment gated on angle ≤5°, offset ≤5px, and overlapping extent — the extent gate is what correctly turns fragmentation into a precision penalty instead of letting it pass silently). Both unit-tested against known cases first.

## Key findings

**Finding 1 — a drive-pier-specific weakness, and a cheap fix.** Stratifying the ResNet34 baseline's results by `has_drive_pier` surfaced a much larger effect than general over-segmentation: line match rate collapsed from **96% (normal rows) to 41% (drive-pier tiles)**, traced to the training `DataLoader` having no stratified sampling for a pattern that's only ~12% of train data. A `WeightedRandomSampler` (5x weight on pier tiles) closed part of the gap with no cost to the majority class, confirmed on held-out test: pier line match rate 64.4%→72.1%, precision 64.1%→72.4%, non-pier unchanged.

**Finding 2 — the transformer backbone outperforms the fix.** Swapping only the encoder (ResNet34→MiT-B0, everything else identical, uniform sampling) beat *both* the ResNet34 baseline and the ResNet34+oversampling fix, with ~4x fewer parameters — confirmed on test:

| (test, non-empty) | ResNet34 baseline | +oversampling | MiT-B0 (no fix needed) |
|---|---|---|---|
| overall line precision | 61.1% | — | **81.7%** |
| pier line match rate | 64.4% | 72.1% | **74.4%** |
| pier line precision | 64.1% | 72.4% | **80.9%** |

Reading: the drive-pier weakness and general fragmentation likely share a root cause — a CNN's limited receptive field producing patchier raw probability masks — and a transformer's global attention addresses that more directly than reweighting the training distribution does. The two aren't mutually exclusive; MiT-B0 + oversampling combined is the natural next experiment.

## Extension and limitations

Implemented row-tilt/misalignment detection (median-deviation flagging) as the brief's optional extension — validated against a synthetic +8° injection (correctly flagged), zero false positives on real farm-wide data (max natural deviation 0.96°). Main limitations: predicted masks are still somewhat fragmented relative to GT even with MiT-B0; fragmentation from real-imagery interruptions vs. drive-pier gaps look like two distinct failure modes (`preliminary_results/vectorize_steps/`) not yet evaluated separately; MiT-B0+oversampling combined is untested; and real-neighbor context-padding (`context_pad.py`, implemented) was left opt-in given the time budget.
