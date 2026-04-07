#!/usr/bin/env python3
"""
Copy RGB frames into datasets/imgs/ and create simple pseudo-label masks in datasets/masks/.

Use when you have extracted AMtown02 frames (e.g. data/extracted_images/images) but no manual labels.
Pseudo masks are grayscale-threshold based — only for bootstrapping / pipeline tests; replace with real labels for final submission.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _pseudo_mask_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Binary mask: bright / sky-like vs darker regions (heuristic, not semantic)."""
    gray = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    t = float(np.median(gray))
    m = (gray > t).astype(np.uint8)
    return m


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare datasets/ from a folder of RGB images + pseudo masks.")
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=REPO_ROOT / "data" / "extracted_images" / "images",
        help="Folder containing source .png frames (VO extraction output).",
    )
    parser.add_argument("--max-files", type=int, default=400, help="Maximum number of frames to copy.")
    parser.add_argument("--glob", type=str, default="*.png", help="Glob pattern under src-dir.")
    args = parser.parse_args()

    if not args.src_dir.is_dir():
        print(f"Source not found: {args.src_dir}")
        print("Run VO extraction first, or pass --src-dir to your frames folder.")
        return 1

    imgs_out = MODULE_ROOT / "datasets" / "imgs"
    masks_out = MODULE_ROOT / "datasets" / "masks"
    imgs_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    files = sorted(args.src_dir.glob(args.glob))[: args.max_files]
    if not files:
        print(f"No files matching {args.glob} in {args.src_dir}")
        return 1

    for idx, src in enumerate(files):
        dst_name = f"frame_{idx:06d}.png"
        dst_img = imgs_out / dst_name
        shutil.copy2(src, dst_img)
        rgb = np.array(Image.open(dst_img).convert("RGB"))
        m = _pseudo_mask_from_rgb(rgb)
        Image.fromarray(m, mode="L").save(masks_out / dst_name)

    print(f"Copied {len(files)} frames to {imgs_out} with pseudo masks in {masks_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
