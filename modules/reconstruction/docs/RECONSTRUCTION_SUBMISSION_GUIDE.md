# AAE5303 - Reconstruction Submission Guide

## 📁 Reconstruction Inputs

The reconstruction module uses COLMAP-format inputs and OpenSplat training.

| Input | Description |
|-------|-------------|
| `baseline/colmap_from_vo_amtown02` | Corrected AMtown02 VO-connected COLMAP project used for all VO OpenSplat runs (**622** registered cameras in `images.bin`) |
| `baseline/colmap_from_amtown02` | **V7** / **方案 B** input: COLMAP baseline on a **292**-view AMtown02 subset; same **aggressive HQ2** OpenSplat recipe as **V6** (quality-ceiling / structured-baseline comparison). This snapshot: **35679** sparse points (VO export here: **50000**)—update counts if your local `images.bin` differs. |

---

## ✅ VO conservative vs aggressive HQ2: same camera count

All VO deliverables **V0–V6** that use `colmap_from_vo_amtown02` share the **same 622 registered cameras** in `images.bin`. The **aggressive HQ2** final (V6) **did not add or remove training images** relative to the conservative sweep; **only OpenSplat hyperparameters** changed.

---

## 📌 Why “more PNG files” does not automatically mean “more OpenSplat training images” (FAQ)

This is **background** for readers who expect the folder PNG count to equal training views. **It is not what differed between conservative and aggressive HQ2 on the VO line** (those share 622 registered views).

OpenSplat only instantiates training cameras from **`images.bin`**. The AMtown02 image folder on disk currently contains on the order of **~7500** PNG files, but the VO COLMAP export lists **622** poses. Until additional frames are reconstructed inside COLMAP (or exported consistently from VO/SLAM into `images.bin`), OpenSplat cannot see them. This limitation is structural to the COLMAP pipeline, not an OpenSplat flag.

---

## 📊 Final Result Structure

Each reconstruction result is described by:

- Input COLMAP project
- OpenSplat profile (**conservative** vs **aggressive HQ2**)
- Iteration count and checkpoint cadence
- Output scene file
- Camera pose count (from `images.bin`)
- Vertex count and file size

### Final selected VO reconstruction result

```json
{
  "result_name": "vo_connected_reconstruction_aggressive_hq2_amtown02",
  "input_source": "colmap_from_vo_amtown02",
  "opensplat_profile": "aggressive_hq2",
  "scene_file": "results/amtown02_vo_amtown02_aggressive_hq2_35000/scene_aggressive_hq2_n35000_d2_sh3.ply",
  "iterations": 35000,
  "checkpoint_interval": 5000,
  "camera_poses": 622,
  "vertices": 1899228,
  "size_mb": 449.19,
  "key_opensplat_flags": "-d 2 --num-downscales 1 --resolution-schedule 1000 --sh-degree 3 --sh-degree-interval 750 --refine-every 50 --warmup-length 300 --densify-grad-thresh 0.00015 -s 5000"
}
```

---

## 📈 VO Iteration Sweep (Conservative Profile)

All runs below use `colmap_from_vo_amtown02` with:

`-d 3 --num-downscales 4 --resolution-schedule 1200 --sh-degree 1`

| Version | Scene file | Iterations | Cameras | Vertices | Size (MB) |
|---------|------------|-----------:|--------:|---------:|----------:|
| V0 | `results/amtown02_vo_amtown02_safe/scene_gpu_safe_n500_d3.ply` | 500 | 622 | 50000 | 4.96 |
| V1 | `results/amtown02_vo_amtown02_10000/scene_gpu_safe_n10000_d3.ply` | 10000 | 622 | 1296361 | 128.58 |
| V2 | `results/amtown02_vo_amtown02_20000/scene_gpu_safe_n20000_d3.ply` | 20000 | 622 | 1338941 | 132.80 |
| V3 | `results/amtown02_vo_amtown02_30000/scene_gpu_safe_n30000_d3.ply` | 30000 | 622 | 1922747 | 190.70 |
| V4 | `results/amtown02_vo_amtown02_35000/scene_gpu_safe_n35000_d3.ply` | 35000 | 622 | 1949324 | 193.34 |
| V5 | `results/amtown02_vo_amtown02_40000/scene_gpu_safe_n40000_d3.ply` | 40000 | 622 | 1807727 | 179.29 |

Use **V6 (aggressive HQ2)** as the repository-facing final VO deliverable; keep **V4** for lightweight browser demos.

---

## 🚀 VO Aggressive HQ2 Final (V6)

| Field | Value |
|-------|-------|
| Scene file | `results/amtown02_vo_amtown02_aggressive_hq2_35000/scene_aggressive_hq2_n35000_d2_sh3.ply` |
| Profile | Aggressive HQ2 (see JSON `key_opensplat_flags` above) |
| Purpose | Higher-frequency detail and stronger SH expressivity than the conservative line |

**Superseded attempt:** matching aggressive knobs with `-d 1` was **killed during image preload** on WSL (memory pressure). The documented recipe therefore uses `-d 2`.

---

## 🧪 **V7** — Changed COLMAP input (Aggressive HQ2)

Same OpenSplat **aggressive HQ2** flag block as **V6** (“次激进” VO recipe: e.g. `-n 35000 -d 2`, `--num-downscales 1`, `--resolution-schedule 1000`, `--sh-degree 3`, `-s 5000`), but OpenSplat reads **`baseline/colmap_from_amtown02`** instead of `colmap_from_vo_amtown02`. **The experimental variable is the COLMAP model** (which cameras and sparse points initialize training). **方案 B** = this baseline-COLMAP HQ2 run for comparison.

| Field | Value |
|-------|-------|
| Label | **V7** |
| Input | `baseline/colmap_from_amtown02` |
| Scene file | `results/amtown02_colmap_baseline_hq2_35000/scene_colmap_baseline_hq2_n35000_d2_sh3.ply` |
| Checkpoints | Same directory: `*_5000.ply`, `*_10000.ply`, … |
| Cameras | 292 (this snapshot) |
| Vertices | 2663956 |
| Size (MB) | 630.06 |

**Not the VO final:** keep **V6** as the VO-connected deliverable. If your local “increased input” build has **>622** `images.bin` entries, still file it under **V7** semantics and refresh the table.

---

## 📊 Quantitative notes (conservative line)

Vertex counts are **not strictly monotonic** after ~35000 because pruning and splat scale edits remove unstable Gaussians even as photometric loss keeps improving. This is why V5 can shrink relative to V4 while still finishing successfully.

---

## 🔢 Lightweight summaries

This module generates:

- `results/RESULT_SUMMARY.md`
- `results/reconstruction_summary.json`

Regenerate with:

```bash
python3 ./scripts/summarize_results.py
```

---

## 📂 Repository Result Files

Keep these under version control:

- `results/RESULT_SUMMARY.md`
- `results/reconstruction_summary.json`
- `results/baseline_report.json`

---

## 📄 Submission Template

Use `final_candidate/submission_template.json` (includes labeled **V0–V7**; **V7** is the changed-input aggressive HQ2 run).
