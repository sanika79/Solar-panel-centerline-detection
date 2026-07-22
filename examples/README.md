# Examples

Five images, each telling a different part of the story — full experimental detail and per-tile numbers are in `preliminary_results/`, this folder is the curated set for a quick walkthrough.

1. **`1_gt_vs_predicted_pier_and_no_pier.png`** — the core result. Ground truth (green) vs. the final model's predicted centerlines (blue) on two held-out test tiles: one normal row, one with a drive-pier gap. Note the predicted line actually breaks at the same pier location as GT, rather than bridging through it.

2. **`2_training_curves.png`** — training dynamics. BCE converges fast and low, but Dice/clDice loss (which specifically measure thin-structure overlap and connectivity) plateau much higher — a quantitative preview of the jagged/fragmented predictions visible in image 1.

3. **`3_drive_pier_gap_found.png`** — the problem, found through stratified evaluation, not assumed: line match rate on drive-pier tiles collapses to 41% vs. 96% for normal rows (val split).

4. **`4_drive_pier_gap_fixed_on_test.png`** — the fix, confirmed on genuinely held-out data. A `WeightedRandomSampler` oversampling drive-pier training tiles 5x closed part of that gap (64%→72% match rate on test), with no meaningful cost to normal-row performance.

5. **`5_misalignment_detection_bonus.png`** — optional extension from the brief: row-tilt estimation for misalignment detection. A synthetically injected +8° tilt (simulating a stuck tracker) is correctly and exclusively flagged against a real-data baseline where natural row-to-row variation never exceeds ~1°.
