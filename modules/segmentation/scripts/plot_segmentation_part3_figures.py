#!/usr/bin/env python3
"""
Generate presentation-quality figures for Part 3 (semantic segmentation) using
real data from:
  - results/uavscenes_training_log.json
  - final_candidate/metrics_best.json

Outputs under figures/part3_presentation/:
  - part3_loss_train_val.png
  - part3_val_metrics_miou_dice_fwiou.png
  - part3_final_metrics_bar.png
  - part3_dashboard_2x2.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODULE_ROOT = Path(__file__).resolve().parents[1]

# PolyU-style maroon + light background (approximate deck theme)
COLOR_MAROON = "#7B2332"
COLOR_MAROON_LIGHT = "#A84858"
COLOR_ACCENT = "#2E5D4E"
BG = "#FFF9F5"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": COLOR_MAROON,
            "axes.labelcolor": "#333333",
            "text.color": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#cccccc",
            "grid.alpha": 0.45,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
        }
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=MODULE_ROOT / "results" / "uavscenes_training_log.json")
    p.add_argument("--metrics", type=Path, default=MODULE_ROOT / "final_candidate" / "metrics_best.json")
    p.add_argument("--out-dir", type=Path, default=MODULE_ROOT / "figures" / "part3_presentation")
    args = p.parse_args()

    if not args.log.is_file():
        raise SystemExit(f"Missing training log: {args.log}")
    rows = json.loads(args.log.read_text(encoding="utf-8"))
    metrics_data: dict = {}
    if args.metrics.is_file():
        metrics_data = json.loads(args.metrics.read_text(encoding="utf-8"))

    epochs = np.array([int(r["epoch"]) for r in rows], dtype=float)
    train_loss = np.array([float(r.get("train_loss", 0.0)) for r in rows])
    val_loss = np.array([float(r.get("val_loss", 0.0)) for r in rows])
    miou = np.array([float(r.get("miou", 0.0)) for r in rows])
    fwiou = np.array([float(r.get("fwiou", 0.0)) for r in rows])
    dice = np.array([float(r.get("dice_score", 0.0)) for r in rows])

    best_idx = int(np.argmax(miou))
    best_ep = int(epochs[best_idx])
    best_miou = float(miou[best_idx])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    # --- Loss ---
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(epochs, train_loss, color=COLOR_MAROON, lw=2.0, label="Train loss")
    ax.plot(epochs, val_loss, color=COLOR_ACCENT, lw=2.0, ls="--", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / validation loss (UAVScenes AMtown02, interval=5)")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True)
    fig.tight_layout()
    p_loss = args.out_dir / "part3_loss_train_val.png"
    fig.savefig(p_loss, dpi=200, facecolor=BG)
    plt.close()

    # --- Val metrics vs epoch ---
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(epochs, miou, color=COLOR_MAROON, lw=2.0, marker="o", ms=3, markevery=max(1, len(epochs) // 20), label="mIoU (val) %")
    ax.plot(epochs, fwiou, color=COLOR_MAROON_LIGHT, lw=1.8, label="fwIoU (val) %")
    ax.plot(epochs, dice, color=COLOR_ACCENT, lw=1.8, label="Dice (val) %")
    ax.axvline(best_ep, color="#999999", ls=":", lw=1.5, label=f"Best val mIoU @ epoch {best_ep}")
    ax.scatter([best_ep], [best_miou], color="black", s=36, zorder=5, label=f"Peak val mIoU = {best_miou:.2f}%")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric (%)")
    ax.set_title("Validation segmentation metrics vs epoch")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True)
    fig.tight_layout()
    p_met = args.out_dir / "part3_val_metrics_miou_dice_fwiou.png"
    fig.savefig(p_met, dpi=200, facecolor=BG)
    plt.close()

    # --- Bar: final test metrics (full-set evaluation) ---
    if metrics_data:
        names = ["mIoU", "Dice", "fwIoU"]
        vals = [
            float(metrics_data["miou"]),
            float(metrics_data["dice_score"]),
            float(metrics_data["fwiou"]),
        ]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        x = np.arange(len(names))
        bars = ax.bar(x, vals, color=[COLOR_MAROON, COLOR_MAROON_LIGHT, COLOR_ACCENT], width=0.55, edgecolor=COLOR_MAROON, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 100)
        pairs = int(metrics_data.get("num_pairs", 0))
        nc = int(metrics_data.get("num_classes", 0))
        ax.set_title(f"Final test metrics (full eval, {pairs} pairs, {nc} classes)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.2f}%", ha="center", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y")
        fig.tight_layout()
        p_bar = args.out_dir / "part3_final_metrics_bar.png"
        fig.savefig(p_bar, dpi=200, facecolor=BG)
        plt.close()
    else:
        p_bar = None

    # --- 2x2 dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax0, ax1, ax2, ax3 = axes.ravel()
    ax0.plot(epochs, train_loss, color=COLOR_MAROON, lw=1.8, label="Train")
    ax0.plot(epochs, val_loss, color=COLOR_ACCENT, lw=1.8, ls="--", label="Val")
    ax0.set_title("Loss")
    ax0.set_xlabel("Epoch")
    ax0.legend(fontsize=8)
    ax0.grid(True)

    ax1.plot(epochs, miou, color=COLOR_MAROON, lw=2, label="mIoU")
    ax1.axvline(best_ep, color="#999999", ls=":", lw=1)
    ax1.set_title("Val mIoU (%)")
    ax1.set_xlabel("Epoch")
    ax1.grid(True)

    ax2.plot(epochs, dice, color=COLOR_MAROON_LIGHT, lw=1.8, label="Dice")
    ax2.plot(epochs, fwiou, color=COLOR_ACCENT, lw=1.8, label="fwIoU")
    ax2.set_title("Val Dice / fwIoU (%)")
    ax2.set_xlabel("Epoch")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    if metrics_data and p_bar:
        names = ["mIoU", "Dice", "fwIoU"]
        vals = [
            float(metrics_data["miou"]),
            float(metrics_data["dice_score"]),
            float(metrics_data["fwiou"]),
        ]
        ax3.bar(names, vals, color=[COLOR_MAROON, COLOR_MAROON_LIGHT, COLOR_ACCENT], edgecolor=COLOR_MAROON)
        ax3.set_ylim(0, 100)
        ax3.set_title("Final test (leaderboard-style)")
        ax3.set_ylabel("%")
        ax3.grid(True, axis="y")
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "Add metrics_best.json", ha="center", va="center")

    fig.suptitle("Part 3 — Semantic segmentation (real run)", fontsize=14, fontweight="bold", color=COLOR_MAROON)
    fig.tight_layout()
    p_dash = args.out_dir / "part3_dashboard_2x2.png"
    fig.savefig(p_dash, dpi=200, facecolor=BG)
    plt.close()

    print(f"Saved figures under: {args.out_dir}")
    print(f"  Best validation mIoU: {best_miou:.4f}% at epoch {best_ep}")
    for path in [p_loss, p_met, p_dash, p_bar]:
        if path:
            print(f"  - {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
