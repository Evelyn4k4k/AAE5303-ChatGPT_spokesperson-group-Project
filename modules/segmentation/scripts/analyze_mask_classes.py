#!/usr/bin/env python3
"""
Analyze unique class ids in a mask folder (UAVScenes).

This is required by the course tips: verify which class IDs actually exist
before choosing n_classes / remapping.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def _load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze unique class IDs in mask PNGs.")
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--glob", type=str, default="*.png")
    p.add_argument("--max-files", type=int, default=0, help="0 = no limit")
    args = p.parse_args()

    files = sorted(args.mask_dir.glob(args.glob))
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        print(f"No files found in {args.mask_dir} with pattern {args.glob}")
        return 1

    counter: Counter[int] = Counter()
    for f in files:
        m = _load_mask(f)
        uniq, counts = np.unique(m, return_counts=True)
        for u, c in zip(uniq.tolist(), counts.tolist(), strict=True):
            counter[int(u)] += int(c)

    ids_sorted = sorted(counter.keys())
    print(f"Files analyzed: {len(files)}")
    print(f"Unique class IDs: {ids_sorted}")
    print("Pixel counts (top 20):")
    for cid, cnt in counter.most_common(20):
        print(f"  {cid}: {cnt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

