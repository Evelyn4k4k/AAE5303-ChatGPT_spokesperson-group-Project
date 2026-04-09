#!/usr/bin/env python3
"""
Remap UAVScenes mask PNGs from original class IDs (0..25 with gaps) into
continuous indices 0..(C-1) for training/evaluation.

Course baseline uses 14 valid (named) classes:
  [0, 1, 2, 3, 5, 6, 13, 14, 15, 16, 17, 19, 20, 24]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def main() -> int:
    p = argparse.ArgumentParser(description="Remap UAVScenes masks to continuous class indices.")
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--valid-class-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 5, 6, 13, 14, 15, 16, 17, 19, 20, 24],
        help="Original class IDs to keep (will be mapped to 0..C-1).",
    )
    p.add_argument("--glob", type=str, default="*.png")
    args = p.parse_args()

    files = sorted(args.in_dir.glob(args.glob))
    if not files:
        print(f"No files found in {args.in_dir} with pattern {args.glob}")
        return 1

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    id_to_idx = {int(cid): i for i, cid in enumerate(args.valid_class_ids)}
    unknown_value = 0  # map unknown/ignored to background index

    for f in files:
        m = _load_mask(f)
        remapped = np.full_like(m, fill_value=unknown_value)
        for cid, idx in id_to_idx.items():
            remapped[m == cid] = idx
        Image.fromarray(remapped.astype(np.uint8), mode="L").save(out / f.name)

    mapping_path = out / "class_mapping.json"
    mapping = {
        "valid_class_ids": [int(x) for x in args.valid_class_ids],
        "id_to_idx": {str(k): int(v) for k, v in id_to_idx.items()},
        "note": "Masks remapped to 0..C-1; unknown ids -> background(0).",
    }
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    print(f"Remapped {len(files)} masks into {out}")
    print(f"Wrote mapping: {mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

