## OpenSplat Tips (Input -> Training -> Result Selection)

This document is a practical checklist for AAE5303 students to avoid common pitfalls when running OpenSplat for 3D scene reconstruction.

---

### 1) Use the corrected AMtown02 COLMAP input

OpenSplat expects:

- image files
- `cameras.bin`
- `images.bin`
- `points3D.bin`

For VO-connected runs, use:

- `baseline/colmap_from_vo_amtown02`

If sparse points are missing, OpenSplat will not produce a valid reconstruction.

---

### 2) Remember that COLMAP, not OpenSplat, decides how many images train

Only cameras listed in `images.bin` participate.

**VO line in this repo:** conservative **V0–V5** and aggressive HQ2 **V6** all use `colmap_from_vo_amtown02` with **622** registered cameras. V6 **did not change** how many images train versus V0–V5—**only OpenSplat flags changed**. **V7** is the aggressive HQ2 recipe again, but on **`colmap_from_amtown02`** (different `images.bin` in this snapshot: **292** cameras)—i.e. a **changed-input** experiment.

**FAQ:** the AMtown02 image directory may list **~7500** PNG files on disk while the VO COLMAP model still lists **622** poses. Extra PNGs do **not** increase OpenSplat coverage until they are registered in COLMAP `images.bin`.

---

### 3) Keep the COLMAP input fixed when comparing iteration counts

All conservative sweeps should reuse the same `colmap_from_vo_amtown02` project so differences isolate OpenSplat iterations and pruning behavior.

---

### 4) Pick a training profile deliberately

**Conservative profile (stable, smaller files, milder SH):**

```bash
./opensplat /path/to/colmap_from_vo_amtown02 \
  -n 35000 \
  -d 3 \
  --num-downscales 4 \
  --resolution-schedule 1200 \
  --sh-degree 1 \
  -o /path/to/output.ply
```

**Aggressive HQ2 profile (selected VO final in this repo, higher detail, larger files):**

```bash
./opensplat /path/to/colmap_from_vo_amtown02 \
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
  -o /path/to/output.ply
```

**Why these knobs differ:** lower `-d` / `--num-downscales` keep training closer to full resolution for more steps; higher `--sh-degree` captures view-dependent color; densification flags adapt Gaussians faster; `-s 5000` emits checkpoints for debugging.

**Memory warning:** an experiment with `-d 1` plus the aggressive block was **killed during preload** on WSL. Stay at `-d 2` unless you have ample host RAM.

---

### 5) Large iteration counts improve density, not input coverage

More iterations often grow or prune Gaussians, but they never add new camera poses. If geometry is bent or incomplete, fix COLMAP / trajectory quality before chasing more OpenSplat steps.

---

### 6) Keep large binaries out of normal GitHub commits

Large `.ply` and `.splat` files should usually remain local or be uploaded through an external file host.

For GitHub, prefer:

- `README.md`
- scripts
- `RESULT_SUMMARY.md`
- `reconstruction_summary.json`
- `baseline_report.json`

---

### 7) Use summaries for repository sharing

To regenerate lightweight summaries:

```bash
python3 ./scripts/summarize_results.py
```

This keeps the GitHub repository readable without uploading all heavy scene files.
