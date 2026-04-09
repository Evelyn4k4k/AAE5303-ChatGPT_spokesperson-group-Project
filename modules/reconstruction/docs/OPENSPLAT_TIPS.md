## OpenSplat Tips (Input -> Training -> Result Selection)

This document is a practical checklist for AAE5303 students to avoid common pitfalls when running OpenSplat for 3D scene reconstruction.

---

### 1) Use the corrected AMtown02 COLMAP input

OpenSplat expects:

- image files
- `cameras.bin`
- `images.bin`
- `points3D.bin`

For the updated GitHub-facing reruns, use:

- `baseline/colmap_from_vo_amtown02`

If sparse points are missing, OpenSplat will not produce a valid reconstruction.

---

### 2) Keep the input fixed during iteration comparison

All corrected reruns should use the same `colmap_from_vo_amtown02` input so that the comparison only reflects iteration changes.

---

### 3) Large iteration counts improve density, not input coverage

Increasing iterations often makes the scene denser, but it does not add new viewpoints.

Typical interpretation:

- More iterations -> denser gaussians
- Better input poses -> better completeness

If your reconstruction is still incomplete, the bottleneck is often the input trajectory rather than the optimizer.

---

### 4) Prefer the stable corrected rerun setting

For this repository, the most stable corrected configuration is:

```bash
./opensplat /path/to/colmap_from_vo_amtown02 \
  -n 35000 \
  -d 3 \
  --num-downscales 4 \
  --resolution-schedule 1200 \
  --sh-degree 1 \
  -o /path/to/output.ply
```

This setting produces the selected corrected AMtown02 result currently used in the repository summary.

---

### 5) Keep large binaries out of normal GitHub commits

Large `.ply` and `.splat` files should usually remain local or be uploaded through an external file host.

For GitHub, prefer:

- `README.md`
- scripts
- `RESULT_SUMMARY.md`
- `reconstruction_summary.json`
- `baseline_report.json`

---

### 6) Use summaries for repository sharing

To regenerate lightweight summaries:

```bash
python3 ./scripts/summarize_results.py
```

This keeps the GitHub repository readable without uploading all heavy scene files.
