# Solar Tracker Centerline Detection

Detects solar-tracker row centerlines in 500x500 px, 4cm-GSD aerial orthomosaic
tiles using a U-Net segmentation model + vectorization pipeline, evaluated
with buffered pixel and line-level matching metrics. Includes a bonus
extension for row-tilt / misalignment detection.

See `report/REPORT.md` for methodology and findings (short version),
`report/METHODOLOGY_WALKTHROUGH.md` for the full detailed writeup, and
`examples/` for curated result visualizations.

## Repo layout

```
solar-centerline/            <- this folder (the code deliverable)
├── src/
│   ├── data/                 preprocessing: manifest, SVG parsing, rasterization,
│   │                         spatial split, tile loading/padding, augmentation, Dataset
│   ├── model/                U-Net, composite loss, training loop, inference
│   ├── postprocess/          mask -> vector line segments
│   ├── eval/                 buffered pixel P/R/F1 + line-level matching metrics
│   └── analysis/             row-tilt / misalignment detection (bonus)
├── scripts/                  CLI entrypoints (see Usage below)
├── tests/                    unit tests, run directly with the venv's python
├── outputs/                  generated: manifest/split CSVs, checkpoints, predictions, metrics
├── preliminary_results/      full experiment log with all intermediate findings
├── examples/                 5 curated result images for a quick walkthrough
└── report/                   REPORT.md (short) + METHODOLOGY_WALKTHROUGH.md (detailed)
```

**Note on the environment**: the working Python venv (`.venv/`) and `pyproject.toml`
for this project live one directory **above** this folder — i.e. in the
parent directory that also contains the `CenterLine_Dataset/` data folder,
not inside `solar-centerline/` itself. All commands below assume your
working directory is that parent folder.

## Setup

From the parent directory (the one containing both `solar-centerline/` and `CenterLine_Dataset/`):

```bash
uv venv                     # only if .venv doesn't already exist here
uv pip install numpy pandas opencv-python scikit-image pillow matplotlib
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # or the CPU-only wheel if no CUDA GPU
uv pip install segmentation-models-pytorch
```

## Data

Point `--data-dir` at the dataset root (the folder containing `tiles/`,
`labels/`, `metadata/`), e.g. `CenterLine_Dataset/CenterLine_Dataset`.

## Usage

Most commands run from the **parent directory** (where `.venv/` lives),
using `.venv/Scripts/python.exe` (Windows) — swap in `.venv/bin/python` on
Linux/Mac. The two exceptions are `manifest.py`/`split.py`, which use
relative imports and must be run with `-m` **from inside `solar-centerline/`**
(noted explicitly below) — every other script/test has an explicit
`sys.path` shim so it works as a direct file path from the parent instead.

```bash
DATA_DIR="CenterLine_Dataset/CenterLine_Dataset"
PY=".venv/Scripts/python.exe"

# 1. Build the enriched tile manifest (is_empty, is_boundary, has_drive_pier, ...)
#    NOTE: run from inside solar-centerline/, with -m (relative imports)
(cd solar-centerline && ../.venv/Scripts/python.exe -m src.data.manifest \
    --data-dir "../$DATA_DIR" --out outputs/manifest/tiles_manifest.csv)

# 2. Compute the spatial (col_idx-block) train/val/test/buffer split
#    NOTE: also run from inside solar-centerline/, with -m
(cd solar-centerline && ../.venv/Scripts/python.exe -m src.data.split \
    --manifest outputs/manifest/tiles_manifest.csv --out outputs/manifest/tiles_split.csv)

# (1+2, plus mask caching and a qualitative preprocessing preview, in one step
#  -- this one runs from the parent dir like everything below, no -m needed:)
$PY solar-centerline/scripts/preprocess.py --data-dir $DATA_DIR \
    --out solar-centerline/outputs

# 3. Train (baseline, uniform sampling)
$PY solar-centerline/src/model/train.py --data-dir $DATA_DIR \
    --manifest solar-centerline/outputs/manifest/tiles_split.csv \
    --out solar-centerline/outputs/checkpoints/main \
    --epochs 30 --loss bce_dice_cldice --thickness 3

# 3b. Train with drive-pier oversampling (the experiment that improved results)
$PY solar-centerline/src/model/train.py --data-dir $DATA_DIR \
    --manifest solar-centerline/outputs/manifest/tiles_split.csv \
    --out solar-centerline/outputs/checkpoints/oversample_pier \
    --epochs 30 --loss bce_dice_cldice --thickness 3 --pier-oversample-weight 5.0

# 4. Predict + vectorize on a split
$PY solar-centerline/scripts/predict.py \
    --checkpoint solar-centerline/outputs/checkpoints/oversample_pier/best.pt \
    --data-dir $DATA_DIR \
    --manifest solar-centerline/outputs/manifest/tiles_split.csv \
    --split test --out solar-centerline/outputs/predictions/test

# 5. Evaluate (buffered pixel P/R/F1 + line-level matching, with is_empty/
#    has_drive_pier/is_boundary breakdowns)
$PY solar-centerline/scripts/evaluate.py --data-dir $DATA_DIR \
    --manifest solar-centerline/outputs/manifest/tiles_split.csv \
    --pred-dir solar-centerline/outputs/predictions/test \
    --split test --out solar-centerline/outputs/metrics_test.csv

# 6. Plot training curves
$PY solar-centerline/scripts/plot_training.py \
    --log solar-centerline/outputs/checkpoints/oversample_pier/train_log.csv \
    --out solar-centerline/preliminary_results/training_curves.png

# 7. Stage-by-stage vectorization walkthrough on one tile (diagnostic)
$PY solar-centerline/scripts/explain_vectorize_steps_individual.py \
    --checkpoint solar-centerline/outputs/checkpoints/oversample_pier/best.pt \
    --data-dir $DATA_DIR --tile-id tile_r4000_c19000 \
    --out-dir solar-centerline/preliminary_results/vectorize_steps

# Tests (no pytest needed, each file is directly runnable)
$PY solar-centerline/tests/test_metrics.py
$PY solar-centerline/tests/test_vectorize.py
$PY solar-centerline/tests/test_augment.py
$PY solar-centerline/tests/test_orientation.py
```

## Outputs

- `outputs/manifest/tiles_manifest.csv`, `tiles_split.csv` — per-tile metadata + split assignment
- `outputs/checkpoints/<run>/best.pt`, `train_log.csv` — model weights + training history
- `outputs/predictions/<split>/*.svg` — predicted centerlines, same schema as the GT labels
- `outputs/metrics_*.csv` — per-tile evaluation results
- `examples/` — 5 curated result images (start here for a quick look)
- `preliminary_results/` — the full experiment log, including the drive-pier gap finding, the oversampling fix, and the misalignment-detection bonus, each with saved metrics/plots and a README explaining them
