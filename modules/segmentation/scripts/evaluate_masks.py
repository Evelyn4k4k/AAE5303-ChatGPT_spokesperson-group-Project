#!/usr/bin/env python3
"""
Compute segmentation metrics for leaderboard-style reporting (percent scale, matching UNet demo keys).

Metrics (on validation pairs with identical basenames):
  - miou: mean IoU across classes (%)
  - fwiou: frequency-weighted IoU (%)
  - dice_score: mean Dice across classes (%)

Assumes integer masks with class ids 0..C-1 for both pred and GT PNGs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def _normalize_labels(arr: np.ndarray, num_classes: int) -> np.ndarray:
    """Map common PNG encodings to class indices 0..C-1 (binary: 0/255 -> 0/1)."""
    arr = np.asarray(arr, dtype=np.int64)
    if num_classes == 2 and arr.max() > 1:
        return (arr > 127).astype(np.int64)
    return arr


def _compute_confusion(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    mask = (gt >= 0) & (gt < num_classes)
    pred = pred[mask]
    gt = gt[mask]
    idx = gt * num_classes + pred
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate predicted masks against ground-truth masks.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Folder with prediction PNGs.")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Folder with ground-truth PNGs.")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--json-out", type=Path, default=None, help="Write metrics JSON for final_candidate/.")
    args = parser.parse_args()

    pred_files = sorted(args.pred_dir.glob("*.png"))
    if not pred_files:
        pred_files = sorted(args.pred_dir.glob("*.jpg"))
    pairs: list[tuple[Path, Path]] = []
    for pp in pred_files:
        stem = pp.stem
        if stem.endswith("_mask"):
            stem = stem[: -len("_mask")]
        gt = args.gt_dir / f"{stem}.png"
        if not gt.is_file():
            gt = args.gt_dir / f"{pp.name}"
        if not gt.is_file():
            logging.warning("Missing GT for %s", pp.name)
            continue
        pairs.append((pp, gt))

    if not pairs:
        logging.error("No matched pred/GT pairs.")
        return 1

    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    for pp, gp in pairs:
        pred = _normalize_labels(_load_mask(pp), args.num_classes)
        gt = _normalize_labels(_load_mask(gp), args.num_classes)
        if pred.shape != gt.shape:
            logging.warning("Shape mismatch %s vs %s — skip", pp.name, gp.name)
            continue
        confusion += _compute_confusion(pred, gt, args.num_classes)

    # IoU per class
    ious = []
    class_counts = []
    for c in range(args.num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        denom = tp + fp + fn
        iou = float(tp / denom) if denom > 0 else float("nan")
        ious.append(iou)
        class_counts.append(int(confusion[c, :].sum()))

    ious_arr = np.array(ious, dtype=np.float64)
    counts = np.array(class_counts, dtype=np.float64)
    valid = np.isfinite(ious_arr)
    miou = float(np.nanmean(ious_arr[valid]) * 100.0)

    total_pixels = counts.sum()
    if total_pixels > 0:
        fwiou = float(np.nansum(ious_arr * counts) / total_pixels * 100.0)
    else:
        fwiou = 0.0

    # Dice from per-class IoU: Dice_c = 2 IoU_c / (1 + IoU_c) only for binary; use standard Dice from confusion
    dices = []
    for c in range(args.num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        dice = float((2 * tp) / denom) if denom > 0 else float("nan")
        dices.append(dice)
    dices_arr = np.array(dices, dtype=np.float64)
    dice_score = float(np.nanmean(dices_arr[np.isfinite(dices_arr)]) * 100.0)

    metrics = {
        "miou": round(miou, 4),
        "fwiou": round(fwiou, 4),
        "dice_score": round(dice_score, 4),
        "num_pairs": len(pairs),
        "num_classes": args.num_classes,
    }
    logging.info("Pairs evaluated: %s", metrics["num_pairs"])
    logging.info("mIoU (%%): %.4f  fwIoU (%%): %.4f  Dice (%%): %.4f", miou, fwiou, dice_score)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logging.info("Wrote %s", args.json_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
