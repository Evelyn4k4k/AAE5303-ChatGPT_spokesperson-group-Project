# AAE5303 - Reconstruction Submission Guide

## 📁 Reconstruction Inputs

The reconstruction module uses COLMAP-format inputs and OpenSplat training.

Current input sources:

| Input | Description |
|-------|-------------|
| `baseline/colmap_from_amtown02` | Richer COLMAP input used as a reference-quality baseline |
| `baseline/colmap_from_vo` | VO-connected COLMAP input used for the final end-to-end reconstruction comparison |

---

## 📊 Final Result Structure

Each final reconstruction result is described by:

- Input source
- Iteration count
- Output scene file
- Camera pose count
- Vertex count
- File size

### Final selected reconstruction result

```json
{
  "result_name": "vo_connected_reconstruction_final",
  "input_source": "colmap_from_vo",
  "scene_file": "results/amtown02_vo_refined_40000_gpu/scene_gpu_safe_n40000_d3.ply",
  "iterations": 40000,
  "camera_poses": 65,
  "vertices": 4798659,
  "size_mb": 475.94
}
```

---

## 📈 Iteration Comparison Used In The Report

The final report should compare the following six VO-connected runs:

| Version | Scene File | Iterations | Camera Poses | Vertices | Size (MB) |
|--------|------------|-----------:|-------------:|---------:|----------:|
| V1 | `results/amtown02_vo_refined_10000_fresh/scene_gpu_safe_n10000_d3.ply` | 10000 | 65 | 1390913 | 137.95 |
| V2 | `results/amtown02_vo_refined_15000/scene_gpu_safe_n15000_d3.ply` | 15000 | 65 | 2081187 | 206.42 |
| V3 | `results/amtown02_vo_refined_20000/scene_gpu_safe_n20000_d3.ply` | 20000 | 65 | 2765440 | 274.28 |
| V4 | `results/amtown02_vo_refined_25000/scene_gpu_safe_n25000_d3.ply` | 25000 | 65 | 3555460 | 352.64 |
| V5 | `results/amtown02_vo_refined_30000_gpu/scene_gpu_safe_n30000_d3.ply` | 30000 | 65 | 4499683 | 446.29 |
| V6 | `results/amtown02_vo_refined_40000_gpu/scene_gpu_safe_n40000_d3.ply` | 40000 | 65 | 4798659 | 475.94 |

Use `V6` as the final selected version in repository-facing documentation.

---

## 📊 Quantitative Trend Analysis

The iteration study shows that the VO-connected reconstruction keeps getting denser as training continues, but the marginal gain gradually decreases.

| Transition | Vertex Gain | Size Gain | Interpretation |
|--------|------------:|----------:|----------------|
| `10000 -> 15000` | 690274 | 68.47 MB | A large gain, showing that 10000 iterations were still under-trained for this VO input. |
| `15000 -> 20000` | 684253 | 67.86 MB | Another strong gain, confirming that the reconstruction still benefits clearly from longer optimization. |
| `20000 -> 25000` | 790020 | 78.36 MB | Density continues to improve substantially and the scene becomes much more complete visually. |
| `25000 -> 30000` | 944223 | 93.65 MB | The largest absolute gain in this sequence, making 30000 a strong practical upgrade over 25000. |
| `30000 -> 40000` | 298976 | 29.65 MB | Still an improvement, but clearly smaller than the earlier jumps, indicating diminishing returns. |

Two patterns are important here:

- The input stays fixed at **65 camera poses**, so the improvement comes from optimization depth rather than extra viewpoints.
- Vertex count keeps increasing, but later iterations mainly improve density and fill-in rather than fundamentally changing scene coverage.

---

## 💡 How to interpret the results

### VO-connected reconstruction result progression

- All comparison runs use the same `colmap_from_vo` input with 65 camera poses.
- The main variable is the total number of OpenSplat iterations.
- Higher iteration counts consistently increase the final number of vertices and scene density.
- The 40000-iteration version is the strongest final result among the compared runs.
- The improvement from 30000 to 40000 is smaller than the earlier gains, so the experiment is stopped at 40000 instead of pushing to a much higher value on the same 4GB GPU.

### Result interpretation

- The selected `40000`-iteration result is the best **VO-connected** reconstruction produced under the current input constraint.
- Its main strength is local density growth across the same fixed set of camera poses.
- Its main limitation is that missing viewpoints in the VO input cannot be recovered purely by running more iterations.
- This is why the result improves steadily from 10000 to 40000, while the scene is still ultimately bounded by the upstream VO coverage.

---

## 🛑 Why The Experiment Stops At 40000

We do not continue beyond 40000 iterations for three practical reasons:

- The gain from 30000 to 40000 is much smaller than the earlier gains from 10000 to 30000, which indicates diminishing returns.
- The output file already grows to **475.94 MB**, making local storage and artifact handling heavier while still not being suitable for direct GitHub upload.
- The project is being run on an **RTX 3050 Ti 4GB**, so 40000 is already close to a reasonable practical limit for stable experimentation on this setup.

---

## 🔢 Lightweight summaries

This module generates two lightweight summary files:

- `results/RESULT_SUMMARY.md`
- `results/reconstruction_summary.json`

These files are recommended for GitHub uploads because they preserve key metrics without committing large binary scene files.

---

## 📂 Repository Result Files

The following lightweight files are intended to stay in the repository:

- `results/RESULT_SUMMARY.md`
- `results/reconstruction_summary.json`
- `results/baseline_report.json`

---

## 📄 Submission Template

Use:

- `final_candidate/submission_template.json`
