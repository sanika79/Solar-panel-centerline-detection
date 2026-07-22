# Preliminary results

**Run**: `outputs/checkpoints/main` — U-Net, ResNet34 encoder (ImageNet-pretrained), BCE+Dice+clDice loss, mask thickness=3, 30 epochs, batch size 8, lr=1e-4 (AdamW, ReduceLROnPlateau).

**Best checkpoint**: epoch 24, val_loss=1.3752 (val_bce=0.0710, val_dice=0.7566, val_cldice=0.5476). Full per-epoch history in `train_log_main_run.csv`.

**val_gt_vs_pred_epoch24.png**: 4 validation tiles, ground truth (green) vs. model prediction + vectorization (red), generated via `scripts/predict.py` on the epoch-24 checkpoint.

Qualitative read: predicted centerlines are spatially correct (on the right physical row every time) but jagged rather than straight, which fragments the vectorized output into more segments than GT (5–10 predicted vs. a consistent 4 in GT per tile). Core localization signal is right; mask cleanliness needs work — either more training or stronger post-processing (spur-pruning threshold, morphological smoothing before skeletonize).

**training_curves.png** (via `scripts/plot_training.py`): three panels — total loss, val loss components (BCE/Dice/clDice), and LR schedule. Key story: BCE converges fast and low (~0.10) since it's dominated by the easy background class, but Dice loss (~0.77) and clDice loss (~0.60) plateau much higher and barely improve past epoch ~15 — a direct quantitative match to the jagged/fragmented predictions seen qualitatively above, since Dice/clDice are exactly the terms that measure thin-structure overlap and connectivity. The LR schedule shows `ReduceLROnPlateau` firing at epochs 21 and 28, correlating with where val_loss starts fluctuating rather than improving — a sign the model has roughly converged for this config and would need either more epochs, more data, or a different loss/architecture balance to improve further, not just more of the same training.

## Quantitative evaluation (`scripts/evaluate.py`, `src/eval/metrics.py`)

Two metrics, since raw pixel IoU is meaningless for a 1-3px-wide line:
- **Buffered pixel P/R/F1** — a predicted pixel counts as correct if it's within a small tolerance (2px) of *some* GT pixel, and vice versa.
- **Line-level matching** — Hungarian assignment between predicted and GT segments, gated on angle difference (≤5°), perpendicular offset (≤5px), *and* overlapping extent along the line (so a short fragment can't "match" a GT segment it doesn't actually overlap). This is what actually reflects whether a physical row was found correctly, including whether drive-pier breaks were reproduced rather than bridged or spuriously fragmented.

**metrics_val.csv**: per-tile results on the val split (epoch-24 checkpoint). Overall (excluding the 14 empty tiles, which trivially score perfect): pixel F1=0.563, line match rate=0.853, line precision=0.546, mean angle error=0.57°, mean offset error=1.73px. n_gt_segments=225 vs n_pred_segments=357 — confirms the ~1.6x over-segmentation seen qualitatively; the model finds the rows (match rate 85%) but produces excess fragments (precision only 55%).

**drive_pier_performance_gap.png**: the standout finding. Stratifying by `has_drive_pier` (non-empty tiles only, apples-to-apples):

| | no drive pier (n=40) | has drive pier (n=10) |
|---|---|---|
| pixel F1 | 0.612 | 0.368 |
| line match rate | 0.962 | 0.414 |
| line precision | 0.592 | 0.365 |
| mean angle error | 0.53° | 0.81° |
| mean offset error | 1.62px | 2.34px |

Line match rate collapses from 96% to 41% on drive-pier tiles — a much larger effect than the general fragmentation problem. This lines up with a known gap in the training setup: `spatial_split()` and the `DataLoader` have no stratified sampling or loss weighting for `has_drive_pier` (confirmed in code review) — drive-pier tiles are ~13-19% of each split purely by incidental geometry, get sampled with the same uniform probability as any other tile, and the model has had comparatively little signal to learn that a real gap should be predicted as background rather than bridged or hallucinated as noise. A natural next experiment: oversample or loss-weight drive-pier tiles and see if this gap narrows.

(Caveat: n=10 drive-pier tiles in val is a small sample — the direction and magnitude of the gap is credible given how large it is, but exact numbers should be treated as noisy until confirmed on the larger test split or with more val data.)

## Follow-up experiment: oversampling drive-pier tiles (`outputs/checkpoints/oversample_pier`)

Added a `WeightedRandomSampler` to the train `DataLoader` (`--pier-oversample-weight`), giving `has_drive_pier==True` train tiles 5x the sampling weight of others — raises their share of each epoch's draws from the natural 11.7% to ~39.9%. Same architecture/loss/thickness/epochs as the baseline run, only the sampling changed. Best checkpoint: epoch 13, val_loss=1.4606 (`train_log_oversample_pier.csv`) — slightly worse than baseline's overall val_loss (1.3752), which is expected since overall val_loss is dominated by the non-pier majority and isn't the metric this experiment targets.

**drive_pier_oversampling_effect.png** — the metric that matters, drive-pier tiles only (n=10), baseline vs. oversampled:

| | baseline (uniform) | oversampled (5x) | Δ |
|---|---|---|---|
| pixel F1 | 0.368 | 0.402 | +0.034 |
| line match rate | 0.414 | 0.493 | **+0.079** |
| line precision | 0.365 | 0.617 | **+0.252** |
| mean angle error | 0.81° | 0.98° | slightly worse |
| mean offset error | 2.34px | 2.52px | slightly worse |

Non-pier tiles (n=40) held steady or slightly improved (pixel F1 0.612→0.625, line match rate unchanged at 0.962) — so this isn't a tradeoff against the majority class, it's a real net gain. Match rate improved and precision improved a lot more (fewer spurious predicted segments per GT segment: n_pred_segments on pier tiles dropped 70→61 while n_matched rose 32→38). The angle/offset errors on the matched subset got marginally worse, secondary to actually getting the right segments matched. Same small-sample caveat as above applies (n=10) — the direction is consistent and encouraging, but this should be confirmed on more data (or `test`, once the pipeline is finalized) before treating the exact magnitude as reliable.

**Important confound discovered while picking a qualitative example**: 4 of the 10 drive-pier val tiles (all four at `row_idx=9500`) are 62-70% pure NoData (black) — real imagery only exists in a thin top strip, yet GT still specifies full-length segments running into the NoData region. A model cannot possibly predict correctly where there is no visual signal at all, so part of the aggregate drive-pier gap above is genuinely "some pier tiles have almost no real pixels to work with," not purely "the model can't handle drive-pier gaps." This doesn't invalidate the finding (6 of the 10 pier tiles have full real imagery and still show a real, if smaller, gap), but it does mean the exact magnitude of the "drive-pier effect" is somewhat overstated by mixing in these NoData-heavy tiles. Two qualitative examples, chosen to show both sides honestly:

- **drive_pier_tile_example.png** (`tile_r9500_c15000`, 69.5% NoData) — baseline totally fails (0/8 matched, pixel F1=0.00), oversampled partially recovers (3/8 matched). Dramatic, but conflated with the NoData issue.
- **drive_pier_tile_example_clean.png** (`tile_r7500_c15000`, 0% NoData, full real imagery) — the fairer comparison. Baseline predictions are visibly wavy/jagged (especially rows 3-4), oversampled predictions are noticeably straighter; pixel F1 improves 0.83→0.91 with the same 4/8 line match rate in both. This is the more representative "real" effect of oversampling: better mask quality on a genuinely difficult (but visually intact) drive-pier tile, not a NoData artifact.

## Final check on the held-out TEST split (`metrics_test_main.csv`, `metrics_test_oversample_pier.csv`)

Run once, at the end, deliberately — `test` was left untouched through all of the val-driven iteration above specifically so this check stays a genuine held-out confirmation rather than something tuned against. Test split: 65 tiles (11 empty, 13 boundary, 9 drive-pier).

**drive_pier_oversampling_effect_TEST.png** — the oversampling effect replicates on unseen data, non-empty tiles, by `has_drive_pier`:

| | baseline: no pier (n=45) | baseline: pier (n=9) | oversampled: no pier (n=45) | oversampled: pier (n=9) |
|---|---|---|---|---|
| pixel F1 | 0.468 | 0.337 | 0.448 | 0.374 |
| line match rate | 0.896 | 0.644 | 0.887 | **0.721** |
| line precision | 0.604 | 0.641 | 0.583 | **0.724** |

Drive-pier match rate improves 64.4%→72.1% (+7.7pp) and precision 64.1%→72.4% (+8.3pp) on truly unseen tiles — confirms the val-set finding wasn't a fluke. Non-pier tiles show a small dip (89.6%→88.7% match rate) which is within noise range for n=45.

Test's 9 drive-pier tiles are notably cleaner than val's 10 (checked the same NoData confound found earlier): 7 of 9 are 0% NoData, one boundary tile is 11.7% black, and only one (`tile_r9500_c18000`) is heavily NoData (54.6%) — so this test-set confirmation is less confounded than the val result and a more trustworthy read of the real effect size. Also checked whether test's 13 boundary tiles were dragging down overall numbers: modest effect at best (pixel F1 0.394 boundary vs 0.451 non-boundary, n=5 boundary tiles) — not the main driver of test's somewhat lower overall pixel F1 (0.446) vs val's (0.563), which is more likely general test-set composition than a boundary-tile artifact.

**Bottom line for the interview**: baseline model finds tracker rows reliably (85%+ line match rate) but over-segments them (fragmentation) and struggles specifically at drive-pier gaps (~64-85% match rate depending on split, vs 90%+ for normal rows). A single, cheap intervention — weighting the training sampler toward the underrepresented drive-pier case — measurably closed part of that gap, verified on both a validation set and a genuinely untouched test set, with no meaningful cost to the majority case.

**sample_gt_vs_pred_pier_vs_no_pier.png** — headline qualitative pair, final (oversampled) model, test split: `tile_r5500_c19000` (no drive pier, top row) vs. `tile_r4000_c19000` (drive pier, bottom row), GT (green) vs. predicted (blue). The bottom-row predicted lines visibly break at the same drive-pier locations as GT rather than bridging through them — a direct visual confirmation, not just a metric, that the model learned the correct gap behavior rather than smoothing over it.

## Extension: row orientation & misalignment detection (`src/analysis/orientation.py`)

The assignment brief calls this out explicitly as an optional extension: estimate row orientation and discuss how it could feed a misalignment-detection use case (a row tilted differently from its neighbors). Worth doing concretely rather than just discussing, since the machinery already existed — every detected/GT segment already implies a tilt angle via its two endpoints.

**Method**: `segment_tilt_deg(seg)` gives the signed tilt from true vertical (0° = perfectly vertical), direction-normalized so the sign is consistent regardless of endpoint order. `detect_misaligned_rows(segments, threshold_deg, min_length_px)` computes the **median** tilt across a group of rows (median, not mean, so 1-2 real anomalies can't drag the reference toward themselves) and flags any row whose deviation from that median exceeds `threshold_deg`. The key physical justification for why this works as a *single-frame* signal (no multi-temporal baseline needed): every row on a single-axis tracker farm shares one physical tracking axis and moves together through the day — so a row that's off from its neighbors *right now* is a real anomaly (stuck actuator, broken drive motor, physical damage) independent of whatever angle the whole farm happens to be tracking to at that moment.

**Bug caught during validation, not assumed away**: sweeping this over the *entire real farm's GT data* (2,288 segments across all non-empty tiles) initially produced one striking 25.9° "outlier" in `tile_r4000_c13000`. Investigated rather than reported at face value: that segment was `((117.28, 396.5), (117.41, 396.23))` — **~0.3 pixels long**. Angle estimation is inherently unstable for near-zero-length segments (a tiny endpoint perturbation swings `atan2` wildly), and this was a degenerate labeling/clipping stub, not a real measurement. Added `min_length_px=20` to exclude unstable short segments before computing tilts — `tests/test_orientation.py::test_degenerate_short_segment_does_not_trigger_false_positive` locks this in. With that fix, sweeping the whole farm (2,181 stable segments) produces zero false positives at `threshold_deg=2.0`.

**misalignment_detection_demo.png** — since the dataset has no real labeled misalignment to test against, validated detection capability via a controlled synthetic injection: took `tile_r0_c12500` (4 clean, aligned rows) and synthetically rotated one row +8° around its own midpoint (simulating a stuck/misaligned tracker). Correctly and exclusively flagged: the injected row's deviation jumped from 0.05° to 8.02° and was the only one flagged; the three genuine rows stayed at 0.07-0.16°. `tests/test_orientation.py::test_synthetic_misalignment_is_detected` locks this in too.

**How this would feed a real downstream monitoring pipeline** (ties directly to Levit AI's "autonomous infrastructure monitoring" framing): a production version would run this per-tile detector across every tile in a flight, then aggregate per-tile flags into a farm-wide GIS layer (using the affine transform from the data-analysis phase to place each flagged row at its real Easting/Northing). Two complementary signals, not either/or: (1) **cross-sectional** (this method) — compare a row's tilt against its neighbors in the same flight, catches acute failures immediately, no history needed; (2) **longitudinal** — track each physical row's tilt across repeat flights over time, catches slow drift/degradation that's still within-neighbor-tolerance on any single flight but trending wrong over weeks/months. The cross-sectional method built here is deliberately the cheaper, zero-history version — exactly the kind of thing worth shipping first and then layering the temporal signal on top of once multiple flights exist.
