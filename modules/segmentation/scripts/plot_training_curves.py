#!/usr/bin/env python3
"""
Plot training curves from modules/segmentation/results/uavscenes_training_log.json.

Outputs:
  - figures/training_loss_curve.png
  - figures/validation_metrics_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

MODULE_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description="Plot loss/mIoU/FWIoU/Dice curves from training log.")
    p.add_argument(
        "--log",
        type=Path,
        default=MODULE_ROOT / "results" / "uavscenes_training_log.json",
        help="Training log JSON path.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=MODULE_ROOT / "figures",
        help="Directory to save curve images.",
    )
    args = p.parse_args()

    if not args.log.is_file():
        raise SystemExit(f"Training log not found: {args.log}")

    rows = json.loads(args.log.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit(f"Training log is empty: {args.log}")

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r.get("train_loss", 0.0)) for r in rows]
    val_loss = [float(r.get("val_loss", 0.0)) for r in rows]
    miou = [float(r.get("miou", 0.0)) for r in rows]
    fwiou = [float(r.get("fwiou", 0.0)) for r in rows]
    dice = [float(r.get("dice_score", 0.0)) for r in rows]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, val_loss, marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    loss_png = out_dir / "training_loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_png, dpi=180)
    plt.close()

    # Validation metrics
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, miou, marker="o", label="mIoU (%)")
    plt.plot(epochs, fwiou, marker="s", label="FWIoU (%)")
    plt.plot(epochs, dice, marker="^", label="Dice (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Metric (%)")
    plt.title("Validation Segmentation Metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    metric_png = out_dir / "validation_metrics_curve.png"
    plt.tight_layout()
    plt.savefig(metric_png, dpi=180)
    plt.close()

    print(f"Saved: {loss_png}")
    print(f"Saved: {metric_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

