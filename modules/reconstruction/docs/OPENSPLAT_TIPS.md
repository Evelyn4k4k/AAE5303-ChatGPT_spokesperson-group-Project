## OpenSplat Tips (Input -> Training -> Result Selection)

This document is a practical checklist for AAE5303 students to avoid common pitfalls when running OpenSplat for 3D scene reconstruction.

---

### 1) Use a valid COLMAP-format input

OpenSplat expects:

- image files
- `cameras.bin`
- `images.bin`
- `points3D.bin`

For this project, the two main inputs are:

- `baseline/colmap_from_amtown02`
- `baseline/colmap_from_vo`

If sparse points are missing, OpenSplat will not produce a valid reconstruction.

---

### 2) Separate quality testing from integration testing

Use:

- `colmap_from_amtown02` to evaluate reconstruction quality
- `colmap_from_vo` to evaluate VO-to-reconstruction integration

Do not expect the VO-connected result to match the richer reconstruction input if the VO pose coverage is much smaller.

---

### 3) Large iteration counts improve density, not input coverage

Increasing iterations often makes the scene denser, but it does not add new viewpoints.

Typical interpretation:

- More iterations -> denser gaussians
- Better input poses -> better completeness

If your reconstruction is still incomplete, the bottleneck is often the input trajectory rather than the optimizer.

---

### 4) Prefer stable parameter settings once validated

For this repository, the most stable VO-connected configuration was:

```bash
./opensplat /path/to/colmap_from_vo \
  -n 25000 \
  -d 3 \
  --num-downscales 4 \
  --resolution-schedule 1200 \
  --sh-degree 1 \
  -o /path/to/output.ply
```

This setting produced the strongest VO-connected result currently stored in the repository.

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
