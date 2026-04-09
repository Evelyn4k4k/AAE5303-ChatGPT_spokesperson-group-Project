# Semantic Segmentation Module (Pytorch-UNet)

This folder contains the **semantic segmentation** part of the AAE5303 group project.

---

## 1. Task Overview

- **Baseline code:** [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- **Target sequence (course context):** **AMtown02** (MARS-LVIG) — align with VO / reconstruction where possible
- **Goal:** reproducible training + hyperparameter tuning, export predictions / metrics, and submit leaderboard-style results

**Leaderboard reference (UNet demo schema):** [AAE5303_UNet_demo / leaderboard](https://github.com/Qian9921/AAE5303_UNet_demo/tree/main/leaderboard)

**Important (real evaluation dataset):** UNet leaderboard uses **UAVScenes AMtown02 (interval=5)** with semantic masks (not just the raw MARS rosbag). See:
- [AAE5303_UNet_demo leaderboard guide](https://github.com/Qian9921/AAE5303_UNet_demo/tree/main/leaderboard)
- [UAVScenes dataset repo](https://github.com/sijieaaa/UAVScenes)

---

## 2. Team Role

- **Evelyn4k4k** — Visual Odometry (`modules/vo/`)
- **wymmust** — 3D Reconstruction (`modules/reconstruction/`)
- **taiwanhaitong-crypto** — Semantic Segmentation (`modules/segmentation/`)

---

## 3. Repository Rules (from course / group)

- Do **not** replace the **root** `README.md` with module-only content.
- Keep module documentation **here** (`modules/segmentation/README.md`).
- Prefer **relative paths** inside this module so others can reproduce your run.
- Add **`Qian9921`** and **`qmohsu`** as collaborators on the private group repo for grading / reproduction.

---

## 4. Folder Structure

```text
modules/segmentation/
├── README.md
├── requirements.txt
├── configs/
│   ├── train_default.yaml
│   └── train_smoke.yaml
│   └── uavscenes_amtown02_interval5_14cls.yaml
├── datasets/
│   └── README.md
├── datasets/imgs/          # git-ignored (you create locally)
├── datasets/masks/         # git-ignored (you create locally)
├── scripts/
│   ├── setup_vendor.py
│   ├── analyze_mask_classes.py
│   ├── remap_uavscenes_masks.py
│   ├── train_uavscenes_unet.py
│   ├── plot_training_curves.py
│   ├── generate_demo_dataset.py
│   ├── prepare_pseudo_from_images.py
│   ├── run_smoke_test.py
│   ├── train_unet.py
│   ├── predict_unet.py
│   ├── evaluate_masks.py
│   └── export_leaderboard_json.py
├── checkpoints/            # git-ignored (trained weights)
├── figures/                # plots / qualitative examples for report
├── results/
│   ├── experiments_template.csv
│   └── predictions/        # git-ignored (batch inference outputs)
└── final_candidate/
    ├── metrics_best.json
    └── ChatGPT_spokesperson_leaderboard.json
```

---

## 5. One-time Setup

From the **repository root**:

```bash
python modules/segmentation/scripts/setup_vendor.py
```

This clones Pytorch-UNet into `modules/segmentation/vendor/Pytorch-UNet` (ignored by Git).

Install Python deps (use a venv if you prefer):

```bash
pip install -r modules/segmentation/requirements.txt
```

### 5.1 Quick smoke test (synthetic data, end-to-end)

From the **repository root**, with `.venv` activated:

```bash
python modules/segmentation/scripts/run_smoke_test.py
```

This will: generate 48 synthetic image/mask pairs → train a few epochs → run inference → write `modules/segmentation/results/smoke_metrics.json`.

- Optional: also refresh leaderboard JSON files (only do this when you intentionally want these numbers in `final_candidate/`):

```bash
python modules/segmentation/scripts/run_smoke_test.py --write-final
```

**Important:** smoke metrics come from **random synthetic** data — they are **not** a valid course submission. Use them only to verify your install. For real results, train on **AMtown02** with proper masks (or a documented labeling pipeline).

### 5.2 AMtown02 frames without manual labels (pseudo masks)

If VO has extracted frames to `data/extracted_images/images/` (see `modules/vo/scripts/data_prep/extract_images_amtown02.py`), you can generate **weak pseudo-labels** for bootstrapping:

```bash
python modules/segmentation/scripts/prepare_pseudo_from_images.py --max-files 400
```

Then tune hyperparameters with `train_unet.py` and `configs/train_default.yaml`. Replace pseudo labels with **real semantic masks** before claiming final segmentation results.

---

## 6. Prepare Data

Read `datasets/README.md`. You need paired files:

- `modules/segmentation/datasets/imgs/`
- `modules/segmentation/datasets/masks/`

**Same basename** per pair (Pytorch-UNet convention). Masks must be consistent with the UNet output:

- `classes: 2` → typical binary segmentation (background / foreground), mask values mapped internally by the dataset loader.

---

## 7. Train (Hyperparameter Tuning)

Default hyperparameters live in `configs/train_default.yaml`. Override on the CLI as needed.

Example:

```bash
python modules/segmentation/scripts/train_unet.py --config modules/segmentation/configs/train_default.yaml
```

Notes:

- Training uses the upstream `train.py` logic, but **wandb is mocked** so you do not need a Weights & Biases account.
- Checkpoints are written to `modules/segmentation/checkpoints/`.

Suggested tuning dimensions (document each run in `results/experiments_template.csv`):

- `learning_rate`, `batch_size`, `epochs`
- `scale` (image downscale factor)
- `amp` on/off
- `classes` (only if your labeling schema changes)

---

## 7.1 UAVScenes AMtown02 interval=5 (14-class baseline workflow)

The course UNet baseline uses **14 valid classes** (exclude unnamed IDs in UAVScenes `cmap.py`) and recommends AdamW + low LR.

### Step A: download UAVScenes interval=5

Get **interval=5** from UAVScenes download links (Google Drive / OneDrive / Baidu / HuggingFace):  
[UAVScenes](https://github.com/sijieaaa/UAVScenes)

You need the **camera images** and **camera semantic labels** for **AMtown02**.

### Step B: place data under this module

Create:

```text
modules/segmentation/datasets/uavscenes_amtown02_interval5/
├── imgs/
└── masks_raw/
```

Put the original UAVScenes mask PNGs into `masks_raw/`.

### Step C: analyze class IDs (sanity check)

```bash
python modules/segmentation/scripts/analyze_mask_classes.py \
  --mask-dir modules/segmentation/datasets/uavscenes_amtown02_interval5/masks_raw
```

### Step D: remap masks to 0..13 (continuous IDs)

```bash
python modules/segmentation/scripts/remap_uavscenes_masks.py \
  --in-dir modules/segmentation/datasets/uavscenes_amtown02_interval5/masks_raw \
  --out-dir modules/segmentation/datasets/uavscenes_amtown02_interval5/masks_remapped_14 \
  --valid-class-ids 0 1 2 3 5 6 13 14 15 16 17 19 20 24
```

### Step E: train with AdamW (recommended)

```bash
python modules/segmentation/scripts/train_uavscenes_unet.py \
  --config modules/segmentation/configs/uavscenes_amtown02_interval5_14cls.yaml
```

This writes checkpoints to `modules/segmentation/checkpoints/` and a log to:

```text
modules/segmentation/results/uavscenes_training_log.json
```

### Step E.1: plot training curves for presentation

```bash
python modules/segmentation/scripts/plot_training_curves.py
```

Outputs:

```text
modules/segmentation/figures/training_loss_curve.png
modules/segmentation/figures/validation_metrics_curve.png
```

### Step F: predict + evaluate + export leaderboard JSON

Use `predict_unet.py` to generate prediction PNGs for the evaluation split, then:

```bash
python modules/segmentation/scripts/evaluate_masks.py \
  --pred-dir <your_pred_dir> \
  --gt-dir modules/segmentation/datasets/uavscenes_amtown02_interval5/masks_remapped_14 \
  --num-classes 14 \
  --json-out modules/segmentation/final_candidate/metrics_best.json

python modules/segmentation/scripts/export_leaderboard_json.py
```

---

## 8. Predict

Example:

```bash
python modules/segmentation/scripts/predict_unet.py ^
  --model modules/segmentation/checkpoints/checkpoint_epoch40.pth ^
  -i path\to\frame0001.png path\to\frame0002.png
```

Outputs default to `modules/segmentation/results/predictions/`.

---

## 9. Evaluate (mIoU / fwIoU / Dice)

This script expects paired prediction vs ground-truth masks with matching names (see `evaluate_masks.py --help`).

Example:

```bash
python modules/segmentation/scripts/evaluate_masks.py ^
  --pred-dir modules/segmentation/results/predictions ^
  --gt-dir modules/segmentation/datasets/masks ^
  --num-classes 2 ^
  --json-out modules/segmentation/final_candidate/metrics_best.json
```

---

## 10. Export Leaderboard JSON

After `metrics_best.json` contains your **real** numbers:

```bash
python modules/segmentation/scripts/export_leaderboard_json.py
```

This writes / refreshes:

- `modules/segmentation/final_candidate/ChatGPT_spokesperson_leaderboard.json`

Schema matches the UNet demo template (`dice_score`, `miou`, `fwiou` in **percent** scale, consistent with `evaluate_masks.py`).

---

## 11. Final Artifacts

Recommended for your report / presentation:

- `figures/` — loss curves (export from your notes), qualitative segmentation overlays
- `final_candidate/metrics_best.json` — best metrics
- `final_candidate/ChatGPT_spokesperson_leaderboard.json` — submission JSON
- `results/experiments_template.csv` — copy to `experiments.csv` and fill in all sweeps

---

## 12. Acknowledgements

- Course: **AAE5303 Robust Control Technology in Low-Altitude Aerial Vehicle**
- Baseline: **Pytorch-UNet** (milesial)

---

## 13. Update Log (for grading / reproducibility)

This section explicitly records engineering updates in this module to help TA/instructor quickly verify work scope.

### 2026-04: Core module scaffolding

- Added full module structure under `modules/segmentation/`:
  - `configs/`, `scripts/`, `datasets/`, `results/`, `figures/`, `final_candidate/`
- Added reproducibility docs and command examples in this README.
- Added local smoke pipeline (`run_smoke_test.py`) for environment sanity checks.

### 2026-04: UAVScenes real-evaluation workflow

- Added `configs/uavscenes_amtown02_interval5_14cls.yaml` for course-aligned setup.
- Added `scripts/analyze_mask_classes.py` to inspect actual label IDs in masks.
- Added `scripts/remap_uavscenes_masks.py` to map sparse original IDs to contiguous 0..13.
- Added `scripts/train_uavscenes_unet.py`:
  - AdamW optimizer (instead of upstream RMSprop default),
  - support for long training + checkpoint logs,
  - `--epochs` override for fast iterative checks.
- Updated `.gitignore` to exclude large local UAVScenes data from commits.

### 2026-04: Training visibility & presentation assets

- Added per-batch tqdm progress visualization in `train_uavscenes_unet.py` (loss shown live).
- Added `scripts/plot_training_curves.py` for automatic figure generation:
  - `figures/training_loss_curve.png`
  - `figures/validation_metrics_curve.png`

### How we document future updates

- For every new script/config:
  1. add path + purpose in this section,
  2. add exact command usage in the corresponding workflow section,
  3. note impact on leaderboard reproducibility (metrics / data / class mapping).
