#!/usr/bin/env python3
"""Create synthetic paired imgs/masks under datasets/ for smoke tests (no external data)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

MODULE_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic demo segmentation pairs.")
    parser.add_argument("--num", type=int, default=48, help="Number of image/mask pairs.")
    parser.add_argument("--size", type=int, default=128, help="Square image side length (pixels).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    imgs = MODULE_ROOT / "datasets" / "imgs"
    masks = MODULE_ROOT / "datasets" / "masks"
    imgs.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    h, w = args.size, args.size
    for i in range(args.num):
        rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        y0 = int(rng.integers(0, h // 3))
        y1 = int(rng.integers(2 * h // 3, h))
        x0 = int(rng.integers(0, w // 3))
        x1 = int(rng.integers(2 * w // 3, w))
        y0, y1 = sorted([y0, y1])
        x0, x1 = sorted([x0, x1])
        if y1 <= y0:
            y1 = y0 + 1
        if x1 <= x0:
            x1 = x0 + 1
        rgb[y0:y1, x0:x1] = rng.integers(30, 90, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 1

        name = f"demo_{i:04d}.png"
        Image.fromarray(rgb).save(imgs / name)
        Image.fromarray(mask, mode="L").save(masks / name)

    print(f"Wrote {args.num} pairs to:\n  {imgs}\n  {masks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
