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

## Objective

The goal of this module is to run and tune an OpenSplat-based reconstruction pipeline on a selected UAV sequence, generate 3D Gaussian scene outputs from COLMAP-format inputs, track the quality progression of the VO-connected reconstruction across higher iteration counts, and summarize the selected final result in a reproducible way.

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
2. Run OpenSplat with a stable training configuration.
3. Export scene files such as `.ply` or `.splat`.
4. Generate summary artifacts for repository sharing.
5. Compare the VO-connected reconstruction across multiple iteration settings.
6. Select the best final reconstruction result for reporting.

---

## Folder Structure

```text
reconstruction/
├── README.md
├── docs/
│   ├── RECONSTRUCTION_SUBMISSION_GUIDE.md
│   └── OPENSPLAT_TIPS.md
├── scripts/
│   ├── run_opensplat.sh
│   └── summarize_results.py
├── final_candidate/
│   ├── submission_template.json
│   └── figures/
├── results/
│   ├── RESULT_SUMMARY.md
│   ├── reconstruction_summary.json
│   ├── baseline_report.json
│   ├── amtown02_vo_amtown02_safe/scene_gpu_safe_n500_d3.ply
│   ├── amtown02_vo_amtown02_10000/scene_gpu_safe_n10000_d3.ply
│   ├── amtown02_vo_amtown02_20000/scene_gpu_safe_n20000_d3.ply
│   ├── amtown02_vo_amtown02_30000/scene_gpu_safe_n30000_d3.ply
│   ├── amtown02_vo_amtown02_35000/scene_gpu_safe_n35000_d3.ply
│   └── amtown02_vo_amtown02_40000/scene_gpu_safe_n40000_d3.ply
├── baseline/
└── third_party/
```

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
```

### 2. Run corrected AMtown02 VO reruns

All GitHub-facing reconstruction results in this module are based on `baseline/colmap_from_vo_amtown02`.

For the corrected reruns, use:

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

## Final Result

### Recommended GitHub / Viewer Result On AMtown02

- **Scene file:** `results/amtown02_vo_amtown02_35000/scene_gpu_safe_n35000_d3.ply`
- **Iterations:** 35000
- **Camera poses:** 622
- **Vertices:** 1949324
- **File size:** 193.34 MB
- **Input source:** `colmap_from_vo_amtown02`
- **Reason selected:** corrected AMtown02 rerun selected as the default GitHub-facing and viewer-facing result

### Longest rerun kept for comparison

- **Scene file:** `results/amtown02_vo_amtown02_40000/scene_gpu_safe_n40000_d3.ply`
- **Iterations:** 40000
- **Camera poses:** 622
- **Vertices:** 1807727
- **File size:** 179.29 MB
- **Status:** completed successfully but may appear blank in PlayCanvas due to extreme scale outliers in a small subset of splats

These values are summarized in:

```text
results/baseline_report.json
results/reconstruction_summary.json
```

---

## Experimental Results Comparison

The following table summarizes the corrected AMtown02 reruns generated from `baseline/colmap_from_vo_amtown02`.

| Version | Input | Output | Iterations | Camera Poses | Vertices | Size (MB) | Notes |
|---|---|---|---:|---:|---:|---:|---|
| V0 | `colmap_from_vo_amtown02` | `scene_gpu_safe_n500_d3.ply` | 500 | 622 | 50000 | 4.96 | Fast smoke test confirming the corrected AMtown02 input and CUDA setup. |
| V1 | `colmap_from_vo_amtown02` | `scene_gpu_safe_n10000_d3.ply` | 10000 | 622 | 1296361 | 128.58 | First substantial corrected rerun after fixing the VO image source. |
| V2 | `colmap_from_vo_amtown02` | `scene_gpu_safe_n20000_d3.ply` | 20000 | 622 | 1338941 | 132.80 | Slightly denser than 10000, but still under the later viewer-friendly run. |
| V3 | `colmap_from_vo_amtown02` | `scene_gpu_safe_n30000_d3.ply` | 30000 | 622 | 1922747 | 190.70 | Strong viewer-friendly rerun kept for comparison. |
| V4 | `colmap_from_vo_amtown02` | `scene_gpu_safe_n35000_d3.ply` | 35000 | 622 | 1949324 | 193.34 | Selected corrected rerun for GitHub and PlayCanvas presentation. |
| V5 | `colmap_from_vo_amtown02` | `scene_gpu_safe_n40000_d3.ply` | 40000 | 622 | 1807727 | 179.29 | Completed successfully, but PlayCanvas may show a blank scene because of a few very large splat scales. |

## Quantitative Trend Analysis

The corrected AMtown02 reruns show that viewer-friendly quality is not strictly monotonic with iteration count.

| Transition | Vertex Gain | Size Gain | Interpretation |
|---|---:|---:|---|
| `500 -> 10000` | +1246361 | +123.62 MB | The corrected input quickly becomes dense once the optimizer is allowed to run. |
| `10000 -> 20000` | +42580 | +4.22 MB | Only a small gain, suggesting that 20000 is not materially different from 10000 for this corrected input. |
| `20000 -> 30000` | +583806 | +57.90 MB | A clear improvement that materially increases scene density. |
| `30000 -> 35000` | +26577 | +2.64 MB | A smaller but positive gain that makes `35000` the selected corrected rerun. |
| `35000 -> 40000` | -141597 | -14.05 MB | Vertex count drops and the viewer becomes unreliable, which makes 40000 less suitable for presentation. |

Two patterns are important here:

1. The input stays fixed at **622 camera poses**, so the differences come from optimization depth rather than extra viewpoints.
2. The best presentation result is not the numerically largest iteration count; `35000` is now the selected default while `40000` remains less suitable for browser viewing.

## Result Interpretation

The selected `35000`-iteration result should be interpreted as the best corrected **VO-connected** reconstruction for repository sharing and browser-based viewing on AMtown02.

In practical terms:

1. It starts from the corrected `colmap_from_vo_amtown02` input used throughout the updated report.
2. It preserves the preferred balance between density, file size, and PlayCanvas compatibility in the current report.
3. The `40000` rerun is still useful as a raw experiment artifact, but not as the primary GitHub-facing result.
4. Missing viewpoints in the VO input still cannot be recovered purely by running more iterations.

## Discussion

Several important observations can be drawn from the corrected reruns:

1. The corrected `colmap_from_vo_amtown02` reruns confirm that the pipeline works on AMtown02 with the intended dataset input.
2. `35000` is the selected GitHub-facing result because it extends the corrected rerun line while remaining the preferred presentation artifact.
3. `40000` finishes successfully but introduces a few very large splat scales, which is a practical viewer compatibility issue even though the file is numerically valid.
4. Lightweight summaries are much more suitable for GitHub than uploading large `.ply` binaries directly.

## Why We Keep 35000 And 40000

We keep both reruns for different purposes:

1. `35000` is the preferred presentation file because it is the selected corrected rerun for repository-facing reporting.
2. `40000` is kept as the longest corrected AMtown02 rerun for raw comparison and debugging.
3. The project is being run on an **RTX 3050 Ti 4GB**, so even these longer reruns are already a substantial local cost.
4. The viewer issue at `40000` suggests that pushing even further is more likely to create splat outliers than a clearly better browser result.

## Result Files In This Module

The most important repository-visible files for this module are:

- `results/baseline_report.json`
- `results/reconstruction_summary.json`
- `results/RESULT_SUMMARY.md`
- `final_candidate/submission_template.json`

These files keep the reconstruction result reproducible and easy to inspect without requiring large binary scene files to be committed into Git history.

## Recommended Final Configuration

Based on the corrected AMtown02 reruns, the recommended final setting is:

- **VO-connected reconstruction:** `colmap_from_vo_amtown02` with 35000 iterations for GitHub / viewer presentation

This version was selected because it best represents:

- The corrected end-to-end AMtown02 pipeline result
- The selected GitHub-facing output among the tested reruns

For module-specific documentation, see:

- `docs/RECONSTRUCTION_SUBMISSION_GUIDE.md`
- `docs/OPENSPLAT_TIPS.md`
- `final_candidate/submission_template.json`

## Key Observations

1. Better input provenance matters before iteration tuning; correcting the AMtown02 image source changed the conclusion of the VO study.
2. The `35000` corrected rerun is the primary result for GitHub-facing reporting in this module.
3. The `40000` corrected rerun is useful as a raw artifact, but it is not the best browser-viewing deliverable.
4. GitHub documentation should focus on summaries, scripts, and metadata rather than large local scene binaries.

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
