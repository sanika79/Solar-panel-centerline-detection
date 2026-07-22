# Solar Tracker Centerline Detection — Methodology Walkthrough

This document walks through every major step, in the order it was actually done, with the reasoning behind each decision. It's written to be defensible in an interview: what was found, what was decided, why, and what was verified rather than assumed. (The formal `<1 page` deliverable report is a condensed version of this — see `report/REPORT.md`.)

---

## 1. Data Analysis

Before writing any preprocessing code, the raw dataset was interrogated directly (via the actual pixel/metadata values, not just the task spec) to understand what it really represented.

### 1.1 What the imagery actually is
670 image tiles (`tiles/*.png`) with matching SVG centerline labels (`labels/*.svg`) and a `metadata/tiles_index.csv` manifest. Visual inspection showed the tiles are aerial crops of a single utility-scale **solar farm**, each tile containing 3-4 parallel single-axis **tracker rows** running roughly top-to-bottom, with visible shadow/tilt geometry.

### 1.2 The tiles form one contiguous mosaic, not independent samples
`tiles_index.csv` has columns `tile_id, row_idx, col_idx, image_path, label_path, xmin, ymin, xmax, ymax`. Cross-checking these fields against each other (not just reading the header) revealed:

- `row_idx` and `col_idx` are **pixel-grid addresses into one full orthomosaic**, not arbitrary tile indices — they match the `tile_rXXXX_cXXXX` filenames exactly and step by 500 (the tile size).
- `col_idx` ranges 0→20,000 (41 possible columns); `row_idx` ranges 0→13,000 (27 possible row-bands). Only 670 of the ~1,107 possible grid cells are populated — the farm's footprint is an irregular polygon (row-band tile counts vary from 8 to 41), and NoData cells were dropped, exactly as the task spec describes.
- The `xmin/ymin/xmax/ymax` columns are **real-world projected coordinates** (Easting/Northing, meters — magnitude ~692k/3,528k is consistent with a UTM zone), not pixel coordinates. Verified adjacent tiles are geographically seamless: `tile_r0_c11000`'s `xmax` is bit-for-bit identical to `tile_r0_c11500`'s `xmin`.
- Critically, `xmin` depends **only** on `col_idx` (confirmed identical across 20 different `row_idx` values at the same `col_idx`), and `ymax` depends **only** on `row_idx`. This means all 670 per-tile bounding boxes are evaluations of **one single global affine transform**, not independently-calibrated per-tile georeferencing.

### 1.3 GPS ↔ pixel affine transform
Derived and verified against the data (not assumed):
```
Easting(col_idx, px)  = 692198.9738 + (col_idx + px) × GSD
Northing(row_idx, py) = 3528884.7107 − (row_idx + py) × GSD
```
where `px, py` are a tile's own local pixel coordinates and `GSD` is the ground sample distance in meters/pixel. Solving from two far-apart same-row tiles gave `GSD = 0.0401067 m/px = 4.0107 cm/px` — matching the task spec's "4 cm GSD" as a rounded description of the true, precisely-measured value. This 6-parameter affine (origin + GSD, no rotation) is exactly what a standard GDAL GeoTransform encodes; the per-tile bounding boxes in the CSV are just that global transform evaluated at each tile's own `(row_idx, col_idx)` offset.

### 1.4 Per-tile metadata analysis: 670 tiles, three distinct categories
Every image was actually opened (not assumed 500×500) and every label actually parsed:

- **657 tiles** are the nominal 500×500 px, footprint 20.0534m × 20.0534m (GSD × 500, consistent with §1.3).
- **23 boundary tiles** are genuinely smaller pixel canvases (e.g. 484×500, or 500×429) — confirmed via the PNG file's own IHDR chunk, not inferred. Important correction made during analysis: a naive `(xmax−xmin)/500` calculation on these tiles suggested a *different* GSD (~3.9cm/px), which would have implied inconsistent georeferencing across the dataset. Reading the actual PNG pixel dimensions resolved this: the GSD is uniform everywhere (§1.3); these 23 tiles are simply smaller canvases (13 clipped in width at `col_idx=20000`, 10 clipped in height at `row_idx=13000` — the farm's true eastern and southern edges), not a different resolution. This distinction matters directly for preprocessing (§2.2).
- **137 tiles** have an empty label (`<svg .../>` with zero `<path>` children) — legitimate background/no-panel tiles (confirmed by viewing the imagery: e.g. `tile_r0_c11000` is mostly black NoData with a small bare-ground corner, no visible rows), not corrupted labels. These must be kept as true negatives, not filtered out (§2.5, §3.2).

### 1.5 Drive piers: a real, labeled structural feature
Inspecting individual multi-segment tiles (e.g. `tile_r11000_c1500`, 8 segments instead of the usual 4) revealed that some physical tracker rows are **split into two segments** by the labeler wherever a drive pier (the motor/gearbox assembly that actuates the tracker) interrupts the row — confirmed by cross-referencing the gap's pixel coordinates against the actual image, where a visible mechanical post sits exactly in that gap. This is a deliberate labeling convention (the gap is a real structural break, not something to smooth over), which shaped both the rasterization approach (§2.3) and became the central finding of the evaluation (§3.5-3.7). A heuristic (`has_drive_pier`, in `manifest.py`) detects this pattern — pairing a segment that reaches the top edge but stops short of the bottom with one that reaches the bottom edge but starts short of the top, at matching x-position — and was validated against 5 hand-checked tiles before trusting it at scale (87/670 tiles flagged).

### 1.6 Row orientation, verified farm-wide (not assumed from a couple of tiles)
Sampling non-empty tiles spread across the *entire* `col_idx` range (0 → 19,500) confirmed every physical tracker row runs **near-vertically** (x-drift of only a few px over the full 500px tile height, everywhere checked) — this is what justified splitting the train/val/test data along `col_idx` rather than `row_idx` (§2.4), since a check on only 1-2 tiles would not have been strong enough evidence for a decision that shapes the entire evaluation methodology.

### 1.7 Other important data-design considerations identified during analysis
- **No pre-defined split existed** in `tiles_index.csv` despite the task spec mentioning one — confirmed by reading the actual header; a spatial split had to be designed from scratch (§2.4).
- **Centerlines routinely continue across tile boundaries** — verified concretely both vertically (`tile_r0_c12500`'s 4 rows end at `y=500` with x-values that match `tile_r500_c12500`'s 4 rows starting at `y=0` to 2 decimal places) and horizontally (`tile_r7000_c12500` exits its right edge at `y=285.77`; `tile_r7000_c13000` enters its left edge at the identical `y=285.77`). This directly motivated the buffer-column mechanism in the split (§2.4) — without it, a physical row could appear on both sides of a train/val boundary.

---

## 2. Data Preprocessing

Everything in `src/data/`, built and unit-tested against real, known tiles before being trusted at scale.

### 2.1 Manifest (`manifest.py`)
Parses `tiles_index.csv` and enriches every row with derived, analysis-driven flags: `is_boundary` (pixel dims ≠ 500×500), `n_segments`, `is_empty`, `has_drive_pier`. This single enriched manifest is the backbone everything downstream (split, dataset, evaluation stratification) reads from — computed once, not re-derived ad hoc in multiple places.

### 2.2 SVG parsing (`svg_io.py`)
Every label is a simple `M x1 y1 L x2 y2` 2-point polyline — confirmed exhaustively, so a regex extract is exact and far cheaper than a general SVG parser. Also includes a writer, in the same schema, for predicted output (so predictions and ground truth are directly comparable file-for-file).

### 2.3 Rasterizing labels into training targets (`rasterize.py`)
Segments are drawn onto a binary mask via `cv2.line`, dilated to a configurable thickness (an ablation axis: 1/3/5px). Each segment is drawn **independently** — the code never merges or bridges nearby segments — which is what makes the drive-pier gap (§1.5) survive automatically into the training target as true background, with no special-case logic required. One real bug caught here: this OpenCV build's own `thickness` parameter does not map 1:1 to actual pixel width (verified: `thickness=3` produced a 5px-wide line). Fixed by drawing at a precise 1px width and dilating separately with an elliptical kernel of the exact requested size — deterministic and version-independent.

### 2.4 Train/val/test split (`split.py`) — column-based, with a buffer
Given §1.6 (rows run vertically, confirmed farm-wide), splitting along `row_idx` would slice essentially every physical row in half between train and test. Splitting along `col_idx` instead cuts **perpendicular** to the rows, so whole physical rows fall on one side of the split. A **buffer** of columns is dropped entirely at each split boundary as a guard band against the rarer case (§1.7) of a row sitting close enough to a column edge to straddle two column-tiles — sized at one full tile-width (500px), verified sufficient by directly measuring the worst-case row drift across the whole dataset (max 10.87px), an order of magnitude smaller than the buffer. Known caveats, found by inspecting the resulting split rather than assumed clean: the split doesn't hit the requested 15%/15% val/test fractions exactly (column-block counts, not tile counts, are what's balanced); val/test never reach the farm's southern extent (`row_idx>9500`) because the farm's irregular footprint means the rightmost columns don't extend as far south as the leftmost ones do; and boundary tiles cluster in `test` because `col_idx=20000` is both the farm's true edge and the highest-`col_idx` block.

### 2.5 Tile shape normalization (`tile_io.py`)
The 23 boundary tiles (§1.4) are edge-padded (not zero-padded) up to a uniform 500×500, with a `valid_mask` marking real vs. synthetic pixels. Edge-pad rather than zero-pad specifically so the model isn't taught a fake "black NoData" class distinct from the real, sparser NoData already present elsewhere in the imagery; `valid_mask` lets loss/metrics later ignore the synthetic strip entirely rather than train or evaluate against fabricated content.

### 2.6 Geometric augmentation (`augment.py`)
The 8 D4 dihedral transforms (0/90/180/270° rotation × optional mirror) are applied to the **segment coordinates**, then the mask is rasterized fresh from the transformed segments — never by interpolating an already-rasterized mask array, which would blur a thin binary target into something no longer a clean {0,1} label. Justified because aerial imagery has no canonical "up," so all 8 are valid without corrupting the geometry. Plus brightness/contrast/synthetic-shadow-band jitter (color-only, no geometry change), matching the real shadow/sun-angle variability seen across tiles (§1.1). One real bug caught here too: the initial continuous-coordinate reflection formula (`size - x`) was off by exactly one pixel from what `np.rot90`/`fliplr` actually do to the array (`size - 1 - x`) — for a thin line this causes a full row/column shift, not a harmless rounding difference. Caught by an exact-match unit test comparing the two approaches on a real tile, not assumed correct from the math alone.

### 2.7 Optional context padding (`context_pad.py`)
Since the tile grid is confirmed geographically contiguous (§1.2), a tile can optionally be padded with **real neighboring-tile pixels** (not mirrored/fabricated ones) so rows truncated at a tile edge get genuine context beyond it. Implemented and tested, but left opt-in rather than used in the main run, given the time budget — a documented "if I had more time" lever rather than a silent omission.

---

## 3. Training & Evaluation

### 3.1 `dataset.py` — tying preprocessing into a PyTorch `Dataset`
`SolarCenterlineDataset` composes every piece above into `__getitem__`: load tile (+`valid_mask`) → \[train split only\] random D4 transform + color jitter → rasterize the mask fresh from the (possibly transformed) segments at the configured thickness → ImageNet-normalize the image → tensors. Augmentation defaults to `train` only — `val`/`test` stay fully deterministic, which matters for trusting the evaluation numbers across repeated runs. Per-tile metadata (`is_empty`, `has_drive_pier`, `is_boundary`) is carried through every sample specifically so results can be stratified later (§3.5) rather than only inspected as one aggregate number.

### 3.2 Split composition actually used
Column-block split (§2.4): train ≈71.2%, val ≈9.6%, test ≈9.7%, buffer ≈9.6% (dropped). Empty tiles (§1.4) are kept in every split, not filtered — the model needs to learn "no row here" as confidently as "row here," and dropping them would both bias training and make evaluation optimistic.

### 3.3 Model and loss
**U-Net with a ResNet34 encoder** (ImageNet-pretrained, via `segmentation_models_pytorch`) — the standard, well-tested default for aerial/satellite semantic segmentation, appropriate given the ~530-tile scale of labeled data. **Loss: composite BCE + Dice + clDice.** clDice (Shit et al., CVPR 2021) was specifically chosen because it's built for thin, tubular/curvilinear targets — it penalizes *broken connectivity* along the predicted centerline in a way plain Dice doesn't (Dice alone tends to just erode thin masks to reduce false positives rather than keep them connected). This also reflects the more current research direction identified during a literature check (hybrid CNN+Transformer architectures with explicit connectivity-aware losses, e.g. Seg-Road, PathMamba) — clDice is the practical, lower-risk way to bring that same idea into a U-Net without betting the whole time budget on an unproven architecture.

### 3.4 Baseline training run
30 epochs, mask thickness=3, batch size 8, AdamW lr=1e-4 with `ReduceLROnPlateau`, on a project-local `uv`-managed environment with CUDA (RTX 3070 Ti). Best checkpoint: epoch 24, val_loss=1.3752. The per-epoch loss-component breakdown told a clear story even before any qualitative inspection: **BCE converges fast and low (~0.10)** — dominated by the easy background class — **while Dice (~0.77) and clDice (~0.60) plateau much higher** and barely improve past epoch ~15. That's a direct quantitative prediction of what the qualitative check then confirmed: predicted centerlines are spatially correct (right row, every time) but jagged and over-segmented (5-10 predicted segments vs. a consistent 4 in GT per tile) — exactly the failure mode Dice/clDice measure and BCE doesn't.

### 3.5 Why these two evaluation metrics
Neither raw pixel IoU nor a naive segment-count comparison would honestly capture this problem:
- **Buffered pixel P/R/F1**: a predicted pixel counts as correct if it's within a small tolerance (τ=2px, implemented via morphological dilation with a `(2τ+1)×(2τ+1)` structuring element) of *some* GT pixel, symmetrically for recall. Raw pixel overlap would score a correctly-drawn line that's shifted by a single pixel as a near-total miss, which is not a meaningful failure for a 1-3px-wide target.
- **Line-level matching**: a Hungarian (one-to-one) assignment between predicted and GT segments, restricted to candidate pairs passing three gates — angle difference ≤5°, perpendicular offset ≤5px, *and* overlapping extent along the line (>0 interval IoU). The extent gate is what makes this metric actually meaningful for this task: without it, angle+offset alone would let a short spurious fragment anywhere along an infinite parallel line falsely "match" a GT segment it doesn't really overlap. This is also precisely what makes *fragmentation* register correctly as a **precision** problem, not get silently absorbed — verified directly with a unit test (one GT segment vs. two fragments spanning its full extent: match rate stays 1.0, precision drops to 0.5, exactly the expected diagnostic).

All of this was validated with unit tests before trusting it on real predictions: perfect-match sanity checks, both-empty edge case (scored as perfect, not an undefined 0/0), angle-gate and offset-gate rejection of wrong-orientation/wrong-row lines, the fragmentation-precision behavior above, and a real drive-pier tile matched against itself (8/8, confirming the drive-pier gap structure round-trips correctly through the whole rasterize → vectorize → match pipeline).

### 3.6 The standout finding: drive-pier tiles are a specific, quantified weakness
Stratifying the baseline's validation results by `has_drive_pier` (non-empty tiles only, apples-to-apples) surfaced a much larger effect than general fragmentation:

| | no drive pier (n=40) | has drive pier (n=10) |
|---|---|---|
| pixel F1 | 0.612 | 0.368 |
| line match rate | 0.962 | 0.414 |
| line precision | 0.592 | 0.365 |

Line match rate collapses from 96% to 41%. Tracing this back to the training setup: `spatial_split()` and the `DataLoader` had **no stratified sampling or loss weighting** for `has_drive_pier` — pier tiles are only ~12% of train, sampled with the same uniform probability as any other tile, giving the model comparatively little signal to learn "predict background here, don't bridge or hallucinate." (One caveat also found and reported honestly: 4 of the 10 val pier tiles are 62-70% NoData, meaning part of the raw gap reflects tiles with almost no real pixels to work with, not purely a modeling weakness — the 6 fully-real pier tiles still show a smaller but real gap on their own.)

### 3.7 Fix: oversampling drive-pier tiles, and the baseline-vs-oversampled comparison
A `WeightedRandomSampler` was added to the train `DataLoader`, giving pier tiles 5x sampling weight — raising their share of each epoch's draws from the natural 11.7% to ~39.9% (`w_i = 5` if `has_drive_pier` else `1`; `P(tile) = w_i / Σw_j`; verified the resulting ~39.9% matches this formula exactly). Same architecture/loss/thickness/epochs as the baseline — only the sampling changed, to isolate the effect of this one intervention. Result, on validation:

| | baseline (uniform) | oversampled (5x) | Δ |
|---|---|---|---|
| pier line match rate | 0.414 | 0.493 | **+0.079** |
| pier line precision | 0.365 | 0.617 | **+0.252** |
| non-pier line match rate | 0.962 | 0.962 | unchanged |

A real gain, not a tradeoff — the non-pier majority class showed no regression. **Confirmed once, deliberately, on the untouched test split** (held out through all val-driven iteration specifically so this check would remain genuine): pier line match rate improved 64.4%→72.1%, precision 64.1%→72.4%, non-pier held steady (89.6%→88.7%, within noise for n=45). Test's pier tiles also happened to be cleaner than val's (only 1 of 9 significantly NoData, vs. 4 of 10 in val), making this test-set confirmation a more trustworthy read of the real effect size than the val result alone.

### 3.8 Encoder ablation: transformer backbone beats the sampling fix

The originally-planned encoder ablation (§ literature-review discussion of SegFormer-style hybrid CNN+Transformer architectures) was actually run: swapped the U-Net's ResNet34 encoder for **MiT-B0** (SegFormer's Mix Vision Transformer, ImageNet-pretrained via `segmentation_models_pytorch`), same loss/thickness/epochs/uniform-sampling as the original baseline, isolating that one variable. MiT-B0 has ~5.5M params vs. ResNet34's ~24M, and converged to a better val_loss (1.3578 at epoch 17) faster than ResNet34 (1.3752 at epoch 24).

Confirmed on held-out test, non-empty tiles: pixel F1 0.446→0.526, line match rate 85.4%→89.4%, and most notably line precision 61.1%→**81.7%**, with predicted segment count dropping from 321 to 246 (GT=220) — substantially less fragmented. Stratified by `has_drive_pier`, MiT-B0 **with no pier-specific intervention at all** beat the ResNet34+oversampling fix: pier line match rate 72.1%→74.4%, pier precision 72.4%→80.9%.

Interpretation: the drive-pier weakness and the general fragmentation problem likely share a root cause (a CNN's limited receptive field producing patchier, less-coherent raw probability masks) that a transformer's global attention addresses more directly than reweighting the training distribution does. This doesn't invalidate the oversampling experiment — it's still a real, cheap, valid fix for a CNN backbone, and the two aren't mutually exclusive (MiT-B0 + oversampling combined is a natural next step) — but it reframes the priority: for this task, architecture choice mattered more than the sampling fix. Qualitative confirmation: `examples/6_gt_vs_predicted_mit_b0_pier_and_no_pier.png` shows the same drive-pier tile whose second row was entirely missing its top segment in the ResNet34+oversampling prediction now fully recovered by MiT-B0.

### 3.9 Other points worth carrying into the writeup
- **Vectorization** (`postprocess/vectorize.py`: threshold → skeletonize → pixel-adjacency graph → prune spurs → fit a line per connected component via SVD/total-least-squares) deliberately never bridges separate connected components — this is what lets the drive-pier gap convention (§1.5, §2.3) survive all the way through to the final predicted output, verified by feeding real ground-truth masks back through the pipeline and confirming 8/8 segments recover for a drive-pier tile (not collapsed to 4).
- **Qualitative confirmation, not just numeric**: on the final (oversampled) model's test-set predictions, a drive-pier tile's predicted lines visibly break at the same location as ground truth rather than bridging through — direct visual evidence the model learned the correct behavior, not just an aggregate metric improvement.
- **Decision to go all-in on a learned (deep-learning) approach**, dropping an initially-planned classical-CV baseline, per direct guidance — the "experimentation" evidence the task brief asks for was instead built from ablations within the learned approach (loss composition, mask thickness, and this drive-pier sampling experiment) rather than a classical-vs-learned comparison.
- **Repo/engineering practices**: modular `src/` layout (`data/`, `model/`, `postprocess/`, `eval/`) with `tests/` validating each component against real, known tiles (not synthetic toy cases only) before trusting it at scale; project-local `uv` environment rather than a global Python install; every numeric claim in this document was computed from the actual pipeline output, and one aggregation mistake (stratifying `has_drive_pier` without first filtering out empty tiles, which silently inflated the non-pier group's score) was caught and corrected before being reported, rather than presented uncritically.
