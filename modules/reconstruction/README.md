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
│   ├── amtown02_vo_refined_10000_fresh/scene_gpu_safe_n10000_d3.ply
│   ├── amtown02_vo_refined_15000/scene_gpu_safe_n15000_d3.ply
│   ├── amtown02_vo_refined_20000/scene_gpu_safe_n20000_d3.ply
│   ├── amtown02_vo_refined_25000/scene_gpu_safe_n25000_d3.ply
│   ├── amtown02_vo_refined_30000_gpu/scene_gpu_safe_n30000_d3.ply
│   └── amtown02_vo_refined_40000_gpu/scene_gpu_safe_n40000_d3.ply
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
./scripts/run_opensplat.sh amtown02_quality
./scripts/run_opensplat.sh amtown02_compact
./scripts/run_opensplat.sh vo_safe
```

### 2. Generate lightweight summaries

```bash
python3 ./scripts/summarize_results.py
```

### 3. Keep summary artifacts

The following files are the main lightweight artifacts tracked in this repository:

- `results/RESULT_SUMMARY.md`
- `results/reconstruction_summary.json`
- `results/baseline_report.json`

---

## Final Result

### Best selected reconstruction result on AMtown02

- **Scene file:** `results/amtown02_vo_refined_40000_gpu/scene_gpu_safe_n40000_d3.ply`
- **Iterations:** 40000
- **Camera poses:** 65
- **Vertices:** 4798659
- **File size:** 475.94 MB
- **Input source:** `colmap_from_vo`
- **Reason selected:** strongest final density among the tested VO-connected runs

These values are summarized in:

```text
results/baseline_report.json
results/reconstruction_summary.json
```

---

## Experimental Results Comparison

The following table summarizes the VO-connected reconstruction progression used for the final comparison in this module.

| Version | Input | Output | Iterations | Camera Poses | Vertices | Size (MB) | Notes |
|---|---|---|---:|---:|---:|---:|---|
| V1 | `colmap_from_vo` | `scene_gpu_safe_n10000_d3.ply` | 10000 | 65 | 1390913 | 137.95 | First clearly dense VO-connected result after switching back to a fresh full run. |
| V2 | `colmap_from_vo` | `scene_gpu_safe_n15000_d3.ply` | 15000 | 65 | 2081187 | 206.42 | Density improves noticeably, but some areas are still limited by pose coverage. |
| V3 | `colmap_from_vo` | `scene_gpu_safe_n20000_d3.ply` | 20000 | 65 | 2765440 | 274.28 | Stronger overall fill-in than 15000, suitable as a near-final version. |
| V4 | `colmap_from_vo` | `scene_gpu_safe_n25000_d3.ply` | 25000 | 65 | 3555460 | 352.64 | Strong previous final version before the 30000 GPU run. |
| V5 | `colmap_from_vo` | `scene_gpu_safe_n30000_d3.ply` | 30000 | 65 | 4499683 | 446.29 | Strong result with a clear gain over 25000 and good cost-performance balance. |
| V6 | `colmap_from_vo` | `scene_gpu_safe_n40000_d3.ply` | 40000 | 65 | 4798659 | 475.94 | Final selected version with the highest density among the compared runs. |

## Quantitative Trend Analysis

The iteration study shows that the VO-connected reconstruction keeps getting denser as training continues, but the marginal gain gradually decreases.

| Transition | Vertex Gain | Size Gain | Interpretation |
|---|---:|---:|---|
| `10000 -> 15000` | +690274 | +68.47 MB | A large gain, showing that 10000 iterations were still under-trained for this VO input. |
| `15000 -> 20000` | +684253 | +67.86 MB | Another strong gain, confirming that the reconstruction still benefits clearly from longer optimization. |
| `20000 -> 25000` | +790020 | +78.36 MB | Density continues to improve substantially and the scene becomes much more complete visually. |
| `25000 -> 30000` | +944223 | +93.65 MB | The largest absolute gain in this sequence, making 30000 a strong practical upgrade over 25000. |
| `30000 -> 40000` | +298976 | +29.65 MB | Still an improvement, but clearly smaller than the earlier jumps, indicating diminishing returns. |

Two patterns are important here:

1. The input stays fixed at **65 camera poses**, so the improvement comes from optimization depth rather than extra viewpoints.
2. Vertex count keeps increasing, but later iterations mainly improve density and fill-in rather than fundamentally changing scene coverage.

## Result Interpretation

The selected `40000`-iteration result should be interpreted as the best **VO-connected** reconstruction produced under the current input constraint, not as the globally best possible reconstruction for AMtown02.

In practical terms:

1. It is the strongest end-to-end result because it starts from the VO pipeline output and still produces a dense final scene.
2. Its main strength is local density growth across the same fixed set of camera poses.
3. Its main limitation is that missing viewpoints in the VO input cannot be recovered purely by running more iterations.
4. This is why the result improves steadily from 10000 to 40000, while the scene is still ultimately bounded by the upstream VO coverage.

## Discussion

Several important observations can be drawn from the experiments:

1. The richer COLMAP input produced the strongest reconstruction quality because it provides better camera coverage and more stable geometry.
2. The VO-connected result successfully demonstrates the end-to-end connection from visual odometry to reconstruction, but its completeness is constrained by the smaller number of available poses.
3. Increasing OpenSplat iterations from 10000 to 40000 consistently improved scene density in the VO-connected runs.
4. The jump from 10000 to 15000 and then to 20000 is substantial, 25000 and 30000 remain strong, and 40000 becomes the best final version among the tested settings.
5. The improvement from 30000 to 40000 is real but noticeably smaller than the earlier gains, which is why the analysis focuses on diminishing returns rather than only on raw vertex growth.
6. Lightweight summaries are much more suitable for GitHub than uploading large `.ply` binaries directly.

## Why We Stop At 40000

We do not continue beyond 40000 iterations for three practical reasons:

1. The gain from 30000 to 40000 is much smaller than the earlier gains from 10000 to 30000, which indicates diminishing returns.
2. The output file already grows to **475.94 MB**, making local storage and artifact handling heavier while still not being suitable for direct GitHub upload.
3. The project is being run on an **RTX 3050 Ti 4GB**, so 40000 is already close to a reasonable practical limit for stable experimentation on this setup.
4. Under the same fixed VO input, running much longer is more likely to enlarge the artifact than to change the scene structure in a meaningful way.

## Result Files In This Module

The most important repository-visible files for this module are:

- `results/baseline_report.json`
- `results/reconstruction_summary.json`
- `results/RESULT_SUMMARY.md`
- `final_candidate/submission_template.json`

These files keep the reconstruction result reproducible and easy to inspect without requiring large binary scene files to be committed into Git history.

## Recommended Final Configuration

Based on the current experiments, the recommended final setting is:

- **VO-connected reconstruction:** `colmap_from_vo` with 40000 iterations

This version was selected because it best represents:

- The final end-to-end group pipeline result
- The strongest density among the tested VO-connected versions

For module-specific documentation, see:

- `docs/RECONSTRUCTION_SUBMISSION_GUIDE.md`
- `docs/OPENSPLAT_TIPS.md`
- `final_candidate/submission_template.json`

## Key Observations

1. Better input camera coverage matters more than simply increasing optimizer iterations.
2. The 40000-iteration VO-connected result is the primary result for final reporting in this module.
3. The comparison should focus on the 10000, 15000, 20000, 25000, 30000, and 40000 iteration versions rather than mixing in the separate richer-input baseline result.
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
