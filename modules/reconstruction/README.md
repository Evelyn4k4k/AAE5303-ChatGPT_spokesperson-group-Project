# 3D Reconstruction Module (OpenSplat) — AAE5303 Group Project

This folder contains the **3D reconstruction** part of our AAE5303 group project for low-altitude aerial vehicle perception.

## Context

According to the course design, the final group project integrates three perception components:

- Visual Odometry
- 3D scene reconstruction
- Semantic segmentation

Our group repo is organized accordingly, and this module focuses on the **3D reconstruction / OpenSplat** part on the **AMtown02** sequence.

## Team

- **Evelyn4k4k** — Visual Odometry
- **wymmust** — 3D Reconstruction (this module)
- **taiwanhaitong** — Semantic Segmentation

## Layout on GitHub

The group repository only tracks **documentation and lightweight metadata** under `modules/reconstruction/`, in the same structure as on the GitHub tree (alphabetical folders after this file):

| Path | Contents |
|------|----------|
| `README.md` | This overview |
| `docs/` | `RECONSTRUCTION_SUBMISSION_GUIDE.md`, `OPENSPLAT_TIPS.md` |
| `final_candidate/` | `submission_template.json`, preview `*.png` |
| `results/` | `RESULT_SUMMARY.md`, `baseline_report.json`, `reconstruction_summary.json` |
| `scripts/` | `run_opensplat.sh`, `summarize_results.py` |

**Not in Git:** COLMAP `baseline/` exports, OpenSplat source/build trees, `.ply` / `.splat` scenes, and per-run `cameras.json` (see root `.gitignore`). Place those locally next to this module when reproducing training.

## Objective

The goal of this module is to run and tune an OpenSplat-based reconstruction pipeline on a selected UAV sequence, generate 3D Gaussian scene outputs from COLMAP-format inputs, sweep iteration counts with a **conservative** training profile, then add an **aggressive HQ2** OpenSplat profile for the final VO-connected deliverable. **V6 (aggressive HQ2 on `colmap_from_vo_amtown02`) changed only OpenSplat hyperparameters** relative to V0–V5; it reused the **same** VO COLMAP project and **622** registered cameras. **V7** documents an additional aggressive HQ2 run that **changes the COLMAP input** (this snapshot: `colmap_from_amtown02`). A FAQ below explains why extra PNG files on disk still do not train more cameras unless COLMAP registration is extended.

This aligns with the course focus on robust spatial perception for low-altitude aerial vehicles, where 3D scene reconstruction is one of the core technical components.

---

## Dataset

- **Sequence**: `AMtown02`
- **Source family**: MARS-LVIG / UAVScenes course pipeline
- **Input modality used in this module**: COLMAP-format sparse reconstruction with image observations

Main external references:

- MARS-LVIG Dataset: `https://mars.hku.hk/dataset.html`
- UAVScenes GitHub: `https://github.com/sijieaaa/UAVScenes`
- OpenSplat: `https://github.com/pierotofy/OpenSplat`

---

## Method

We use **OpenSplat** as the 3D Gaussian scene reconstruction backend.

### Main pipeline

1. Prepare a valid COLMAP-format input.
2. Run OpenSplat with a documented training profile (**conservative** for sweeps, **aggressive HQ2** for the final VO deliverable).
3. Export scene files such as `.ply` or `.splat`.
4. Generate summary artifacts for repository sharing.
5. Compare the VO-connected reconstruction across multiple iteration settings.
6. Select the best final reconstruction result for reporting.

### Version roadmap (labels used in this repo)

| Label | COLMAP input | Registered cameras (`images.bin`) | OpenSplat profile | Main output folder |
|-------|--------------|-----------------------------------|-------------------|--------------------|
| **V0–V5** | `baseline/colmap_from_vo_amtown02` | 622 | Conservative (`-d 3`, `--sh-degree 1`, …) | `amtown02_vo_amtown02_*` |
| **V6** | `baseline/colmap_from_vo_amtown02` | 622 | Aggressive HQ2 (same flags as V7 training recipe) | `amtown02_vo_amtown02_aggressive_hq2_35000` |
| **V7** | **Different** COLMAP project (`baseline/colmap_from_amtown02` in this snapshot) | **292** in this snapshot | Aggressive HQ2 (identical flag block to V6) | `amtown02_colmap_baseline_hq2_35000` |

**V7 meaning:** the controlled change is **the COLMAP model** (which images are registered and which sparse geometry initializes Gaussians), **not** another tweak to the V6 OpenSplat hyperparameters. If you rebuilt COLMAP with **more than 622** registered views and trained with the same aggressive HQ2 recipe, that run still belongs in the **V7 class**—update the `baseline/…` path, `results/…` path, and camera counts in the tables below.

---

## Folder structure

**Tracked on Git** (matches the tree under `modules/reconstruction/` on GitHub): see [Layout on GitHub](#layout-on-github) above.

**Typical local checkout** (large assets and builds are gitignored; create them beside the tracked tree):

```text
modules/reconstruction/
├── README.md, docs/, scripts/, final_candidate/, results/   ← as in Git
├── baseline/                    ← COLMAP projects (local only)
├── opensplat_cpu_src/           ← OpenSplat build tree (local only)
├── third_party/                 ← optional local cmake shims (local only)
└── results/
    ├── RESULT_SUMMARY.md, *.json
    └── amtown02_*/                ← per-run folders: .ply, cameras.json, checkpoints (local)
```

After training, rerun `./scripts/summarize_results.py` so `RESULT_SUMMARY.md` and `reconstruction_summary.json` reflect files on disk.

---

## Environment

Typical local environment used in this project:

- Ubuntu / WSL2
- Python 3.10+
- OpenSplat built locally
- CUDA-enabled GPU setup for accelerated training

---

## Reproducibility

### 1. Run OpenSplat presets

From this module directory:

```bash
./scripts/run_opensplat.sh corrected_safe
./scripts/run_opensplat.sh corrected_30000
./scripts/run_opensplat.sh corrected_35000
./scripts/run_opensplat.sh corrected_40000
./scripts/run_opensplat.sh vo_aggressive_hq2_35000
```

### 2. Run corrected AMtown02 VO reruns

All VO-connected reconstruction lines start from `baseline/colmap_from_vo_amtown02`.

**Conservative profile (V0-V5, PlayCanvas-friendly, smaller files):**

The `LD_LIBRARY_PATH` / `LD_PRELOAD` lines below match **one WSL2 + LibTorch CUDA layout**; replace with your own library paths if different.

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/wym/libtorch-cu118-cxx11/lib:${LD_LIBRARY_PATH:-}"
unset CUBLAS_WORKSPACE_CONFIG
export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD="/home/wym/libtorch-cu118-cxx11/lib/libcublas-3b81d170.so.11:/home/wym/libtorch-cu118-cxx11/lib/libcublasLt-b6d14a74.so.11:/home/wym/libtorch-cu118-cxx11/lib/libcudart-d0da41ae.so.11.0"

./opensplat_cpu_src/build_cuda/opensplat \
  ./baseline/colmap_from_vo_amtown02 \
  -n 35000 \
  -d 3 \
  --num-downscales 4 \
  --resolution-schedule 1200 \
  --sh-degree 1 \
  -o ./results/amtown02_vo_amtown02_35000/scene_gpu_safe_n35000_d3.ply
```

**Aggressive HQ2 profile (selected final, higher detail, larger files):**

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/wym/libtorch-cu118-cxx11/lib:${LD_LIBRARY_PATH:-}"
unset CUBLAS_WORKSPACE_CONFIG
export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD="/home/wym/libtorch-cu118-cxx11/lib/libcublas-3b81d170.so.11:/home/wym/libtorch-cu118-cxx11/lib/libcublasLt-b6d14a74.so.11:/home/wym/libtorch-cu118-cxx11/lib/libcudart-d0da41ae.so.11.0"

./opensplat_cpu_src/build_cuda/opensplat \
  ./baseline/colmap_from_vo_amtown02 \
  -n 35000 \
  -d 2 \
  --num-downscales 1 \
  --resolution-schedule 1000 \
  --sh-degree 3 \
  --sh-degree-interval 750 \
  --refine-every 50 \
  --warmup-length 300 \
  --densify-grad-thresh 0.00015 \
  -s 5000 \
  -o ./results/amtown02_vo_amtown02_aggressive_hq2_35000/scene_aggressive_hq2_n35000_d2_sh3.ply
```

### 3. Generate lightweight summaries

```bash
python3 ./scripts/summarize_results.py
```

### 4. Keep summary artifacts

The following files are the main lightweight artifacts tracked in this repository:

- `results/RESULT_SUMMARY.md`
- `results/reconstruction_summary.json`
- `results/baseline_report.json`

---

## Registered Cameras vs PNG Files on Disk (FAQ, not the V6 experiment variable)

**Scope:** This explains a common confusion about dataset folders vs COLMAP. **It does not describe what changed between conservative V0–V5 and aggressive HQ2 V6:** those VO runs all share the **same 622-view** `colmap_from_vo_amtown02` input; only OpenSplat flags differ.

OpenSplat reads COLMAP projects as **four coupled artifacts** (`cameras.bin`, `images.bin`, `points3D.bin`, plus the image folder). Only rows inside `images.bin` become training cameras.

For AMtown02 the `images` directory currently holds on the order of **~7500** PNG files, while `baseline/colmap_from_vo_amtown02` registers **622** cameras. Simply exporting more PNGs from the bag **does not increase OpenSplat coverage** until those frames are added to the COLMAP model (feature matching, triangulation, bundle adjustment, or a consistent export from VO/SLAM). This is why “use all pictures” is not a flag inside OpenSplat—it is a **COLMAP graph construction** problem.

## OpenSplat Training Profiles

| Profile | When to use | Key flags | Tradeoff |
|---------|-------------|-----------|----------|
| **Conservative** | Fast iteration sweeps, PlayCanvas-friendly `.ply` sizes | `-d 3 --num-downscales 4 --resolution-schedule 1200 --sh-degree 1` | Lower apparent sharpness, very stable on 4GB GPUs |
| **Aggressive HQ2** | Final high-detail VO deliverable | `-d 2 --num-downscales 1 --resolution-schedule 1000 --sh-degree 3 --sh-degree-interval 750 --refine-every 50 --warmup-length 300 --densify-grad-thresh 0.00015 -s 5000` | Larger files, longer runs, more floaters if poses are weak |

**Why the flags changed:** lowering `-d` and `--num-downscales` keeps training closer to native resolution for more steps; raising `--sh-degree` models richer view-dependent color; smaller `--refine-every` plus a lower `--densify-grad-thresh` adapt Gaussians more aggressively; `-s 5000` emits intermediate checkpoints for debugging.

**Failed superseded attempt:** an even more extreme run reused the aggressive knobs but set `-d 1`. The process was **killed during image preload on WSL** (host RAM pressure) before CUDA training began, so the reproducible aggressive recipe stays at `-d 2`.

## Conservative vs aggressive HQ2 on the VO line (same input)

**Controlled variable:** OpenSplat hyperparameters only.

**Held fixed:** `baseline/colmap_from_vo_amtown02`, **622** registered cameras in `images.bin`, **50000** sparse points, and the same on-disk image symlink layout.

**Therefore:** V6 is not “more images than V0–V5”; it is the same poses and observations trained with a sharper, higher-cost OpenSplat preset.

## VO-Connected Iteration Sweep (Conservative Profile)

All rows below share `baseline/colmap_from_vo_amtown02`, **622** registered cameras, and the conservative flag block.

| Version | Output (under `results/…`) | Iterations | Vertices | Size (MB) | Notes |
|---------|----------------------------|-----------:|---------:|----------:|-------|
| V0 | `amtown02_vo_amtown02_safe/scene_gpu_safe_n500_d3.ply` | 500 | 50000 | 4.96 | CUDA and path smoke test |
| V1 | `amtown02_vo_amtown02_10000/scene_gpu_safe_n10000_d3.ply` | 10000 | 1296361 | 128.58 | First dense conservative rerun |
| V2 | `amtown02_vo_amtown02_20000/scene_gpu_safe_n20000_d3.ply` | 20000 | 1338941 | 132.80 | Small gain over V1 |
| V3 | `amtown02_vo_amtown02_30000/scene_gpu_safe_n30000_d3.ply` | 30000 | 1922747 | 190.70 | Strong conservative balance |
| V4 | `amtown02_vo_amtown02_35000/scene_gpu_safe_n35000_d3.ply` | 35000 | 1949324 | 193.34 | Best conservative compromise before outliers grow |
| V5 | `amtown02_vo_amtown02_40000/scene_gpu_safe_n40000_d3.ply` | 40000 | 1807727 | 179.29 | Longest conservative run; PlayCanvas may blank due to splat scale outliers |

### Conservative quantitative transitions

| Transition | Vertex Δ | Size Δ | Takeaway |
|------------|---------:|-------:|----------|
| V0 → V1 | +1246361 | +123.62 MB | Optimizer quickly densifies Gaussians |
| V1 → V2 | +42580 | +4.22 MB | Diminishing returns |
| V2 → V3 | +583806 | +57.90 MB | Major perceived quality jump |
| V3 → V4 | +26577 | +2.64 MB | Incremental polish |
| V4 → V5 | −141597 | −14.05 MB | Pruning + instability for generic viewers |

## Aggressive HQ2 VO Run — **V6** (Selected Final)

- **Scene file:** `results/amtown02_vo_amtown02_aggressive_hq2_35000/scene_aggressive_hq2_n35000_d2_sh3.ply`
- **Input:** `baseline/colmap_from_vo_amtown02` (**622** poses, **50000** sparse points)—**identical registration to conservative V0–V5; only OpenSplat parameters changed**
- **Iterations:** 35000 (checkpoints every 5000 steps via `-s 5000`)
- **Vertices:** 1899228
- **File size:** 449.19 MB
- **Role:** primary GitHub-facing VO reconstruction emphasizing **detail over minimal file size**

Keep `results/amtown02_vo_amtown02_35000/scene_gpu_safe_n35000_d3.ply` when you need a **smaller** conservative reference for browser demos.

## Changed COLMAP input — **V7** (Aggressive HQ2, not VO `622` graph)

**V7** reuses the **same aggressive HQ2 OpenSplat command line as V6** but points OpenSplat at a **different COLMAP project** so the **input** (registered cameras + sparse points) changes.

**Plan B (方案 B) framing:** train **`baseline/colmap_from_amtown02`** with the **identical “次激进” HQ2 OpenSplat flags as V6** (`-n 35000 -d 2`, `--num-downscales 1`, `--resolution-schedule 1000`, `--sh-degree 3`, matching refine/warmup/densify knobs, **`-s 5000`** checkpoints). This is a **COLMAP-structured high-quality baseline** on a **292-view subset** of the same AMtown02 image pool—useful as a **quality-ceiling style** comparison against the VO graph, while **V6** stays the VO-connected deliverable.

**This repository snapshot**

- **COLMAP project:** `baseline/colmap_from_amtown02`
- **Registered cameras:** **292** (`images.bin`)
- **Sparse points:** **35679** (`points3D.bin`) — VO export in this repo lists **50000** sparse points; interpret “better COLMAP geometry” by **registration / coverage / reprojection**, not raw `points3D` count alone.
- **Scene file:** `results/amtown02_colmap_baseline_hq2_35000/scene_colmap_baseline_hq2_n35000_d2_sh3.ply`
- **Checkpoints:** `*_5000.ply`, `*_10000.ply`, … in the same result directory (every **5000** steps).
- **Vertices / size:** ~2 663 956 / ~630 MB

**Not the selected VO deliverable:** V6 remains the course-facing VO-connected final on `colmap_from_vo_amtown02` (622 cameras).

**If your “V7” locally uses an expanded VO/COLMAP export with >622 registered images**, keep the **V7 label** for “changed-input aggressive HQ2” but replace the paths and refresh camera/point counts in `README.md`, `baseline_report.json`, and `submission_template.json`.

## Final Result Summary

| Role | File |
|------|------|
| **Selected VO final (aggressive HQ2)** | `results/amtown02_vo_amtown02_aggressive_hq2_35000/scene_aggressive_hq2_n35000_d2_sh3.ply` |
| Conservative reference (PlayCanvas-friendly) | `results/amtown02_vo_amtown02_35000/scene_gpu_safe_n35000_d3.ply` |
| Longest conservative artifact | `results/amtown02_vo_amtown02_40000/scene_gpu_safe_n40000_d3.ply` |
| **V7** (changed COLMAP input, aggressive HQ2) | `results/amtown02_colmap_baseline_hq2_35000/scene_colmap_baseline_hq2_n35000_d2_sh3.ply` |

Machine-readable metrics live in:

```text
results/baseline_report.json
results/reconstruction_summary.json
```

## Result Files In This Module

The most important repository-visible files for this module are:

- `results/baseline_report.json`
- `results/reconstruction_summary.json`
- `results/RESULT_SUMMARY.md`
- `final_candidate/submission_template.json`

These files keep the reconstruction result reproducible and easy to inspect without requiring large binary scene files to be committed into Git history.

## Recommended Final Configuration

- **VO deliverable:** train `baseline/colmap_from_vo_amtown02` with the **aggressive HQ2** recipe for `35000` iterations (`./scripts/run_opensplat.sh vo_aggressive_hq2_35000`).
- **Lightweight demos:** keep the **conservative** `35000` or `30000` artifacts for smaller downloads and predictable browser viewers.

For module-specific documentation, see:

- `docs/RECONSTRUCTION_SUBMISSION_GUIDE.md`
- `docs/OPENSPLAT_TIPS.md`
- `final_candidate/submission_template.json`

## Key Observations

1. **VO aggressive HQ2 (V6) vs conservative (V0–V5):** same **622** `images.bin` cameras and same COLMAP project—**parameter-only** comparison.
2. **V7** changes the **COLMAP input** while keeping the **aggressive HQ2** recipe; in this repo that is `colmap_from_amtown02` (**292** registered cameras in the checked-in snapshot). If you register **more** views in a new COLMAP export, still treat that as **V7-class** and update the documented paths and counts.
3. Conservative sweeps (V0–V5) isolate iteration effects while holding poses fixed at 622 cameras.
4. Aggressive HQ2 (V6) is the selected final VO output because it pushes sharpness and SH capacity at the cost of file size and training time.
5. A `-d 1` aggressive attempt was killed during preload on WSL; `-d 2` is the reproducible compromise on the project's 4GB GPU setup.
6. Extra PNG files on disk do nothing for OpenSplat until they are registered inside COLMAP `images.bin` (general FAQ; not the variable that changed for V6).
7. GitHub documentation should continue to emphasize summaries and JSON metadata rather than committing multi-hundred-megabyte binaries.

---

## Notes on Course Integration

The AAE5303 project pipeline progresses from:

- **UAV trajectory / pose estimation**
- **3D mapping**
- **semantic segmentation**
- **higher-level navigation / decision making**

This reconstruction module corresponds to the 3D mapping stage of that perception stack.

---

## Acknowledgements

- Course: **AAE5303 Robust Control Technology in Low-Altitude Aerial Vehicle**
- Instructor: **Dr Li-Ta Hsu**
- Reconstruction baseline: **OpenSplat**
