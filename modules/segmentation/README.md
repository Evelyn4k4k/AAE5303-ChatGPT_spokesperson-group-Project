# Semantic Segmentation Module (Pytorch-UNet)

This folder contains the **semantic segmentation** part of the AAE5303 group project.

---

## 1. Task Overview

- **Baseline code:** [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- **Target sequence (course context):** **AMtown02** (MARS-LVIG) — align with VO / reconstruction where possible
- **Goal:** reproducible training + hyperparameter tuning, export predictions / metrics, and submit leaderboard-style results

**Leaderboard reference (UNet demo schema):** [AAE5303_UNet_demo / leaderboard](https://github.com/Qian9921/AAE5303_UNet_demo/tree/main/leaderboard)

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
├── datasets/
│   └── README.md
├── datasets/imgs/          # git-ignored (you create locally)
├── datasets/masks/         # git-ignored (you create locally)
├── scripts/
│   ├── setup_vendor.py
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
