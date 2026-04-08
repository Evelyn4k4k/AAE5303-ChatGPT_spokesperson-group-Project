# Datasets (large files stay local)

Create paired folders **under this directory** (same basenames in both):

```text
datasets/
├── imgs/
└── masks/
```

Training scripts resolve these paths relative to `modules/segmentation/`.

For **AMtown02**, export RGB frames from the MARS rosbag and prepare masks (your own labels or a generated pseudo-mask pipeline). The VO module may already extract images — align filenames with your segmentation masks.

The `imgs/` and `masks/` folders are Git-ignored to avoid committing large binaries.
