#!/usr/bin/env python3
"""
Train UNet on UAVScenes-style masks (already remapped to 0..C-1) with AdamW.

Why this script exists:
- Upstream milesial/Pytorch-UNet train.py uses RMSprop by default.
- Course tips recommend AdamW + lr=5e-5, scale=0.25, BS=1 for UAVScenes AMtown02.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]
VENDOR = MODULE_ROOT / "vendor" / "Pytorch-UNet"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _dice_per_class_from_confusion(conf: np.ndarray) -> list[float]:
    dices: list[float] = []
    c = conf.shape[0]
    for i in range(c):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        dices.append(0.0 if denom <= 0 else float((2 * tp) / denom))
    return dices


@torch.no_grad()
def evaluate_epoch(model, loader, device, num_classes: int, amp: bool) -> dict:
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    n_batches = 0
    ce = nn.CrossEntropyLoss()

    for batch in loader:
        images = batch["image"].to(device=device, dtype=torch.float32)
        targets = batch["mask"].to(device=device, dtype=torch.long)
        with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
            logits = model(images)
            loss = ce(logits, targets)
        total_loss += float(loss.item())
        n_batches += 1

        pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        gt = targets.detach().cpu().numpy().astype(np.int64)
        for p, g in zip(pred, gt, strict=True):
            mask = (g >= 0) & (g < num_classes)
            idx = g[mask] * num_classes + p[mask]
            conf += np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    intersection = np.diag(conf).astype(np.float64)
    union = conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf)
    iou = intersection / (union + 1e-10)
    valid = conf.sum(axis=1) > 0
    miou = float(np.mean(iou[valid])) if np.any(valid) else 0.0

    freq = conf.sum(axis=1) / (conf.sum() + 1e-10)
    fwiou = float((freq[freq > 0] * iou[freq > 0]).sum())

    dices = _dice_per_class_from_confusion(conf)
    mdice = float(np.mean([d for d in dices if d > 0.0])) if dices else 0.0

    return {
        "val_loss": 0.0 if n_batches == 0 else total_loss / n_batches,
        "dice_score": mdice * 100.0,
        "miou": miou * 100.0,
        "fwiou": fwiou * 100.0,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Train UNet on UAVScenes masks (remapped) with AdamW.")
    p.add_argument("--config", type=Path, default=MODULE_ROOT / "configs" / "uavscenes_amtown02_interval5_14cls.yaml")
    p.add_argument("--save-dir", type=Path, default=MODULE_ROOT / "checkpoints")
    p.add_argument("--val-percent", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not (VENDOR / "unet").is_dir():
        logging.error("Vendor code missing. Run: python modules/segmentation/scripts/setup_vendor.py")
        return 1

    cfg = _load_yaml(args.config)
    dataset_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    imgs_dir = (MODULE_ROOT / dataset_cfg.get("imgs_dir", "")).resolve()
    masks_dir = (MODULE_ROOT / dataset_cfg.get("masks_dir", "")).resolve()
    if not imgs_dir.is_dir() or not masks_dir.is_dir():
        logging.error("Dataset folders missing. imgs=%s masks=%s", imgs_dir, masks_dir)
        return 1

    epochs = int(train_cfg.get("epochs", 80))
    batch_size = int(train_cfg.get("batch_size", 1))
    lr = float(train_cfg.get("learning_rate", 5e-5))
    wd = float(train_cfg.get("weight_decay", 1e-8))
    scale = float(train_cfg.get("scale", 0.25))
    amp = bool(train_cfg.get("amp", False))
    val_percent = float(args.val_percent if args.val_percent is not None else 10.0)

    n_channels = int(model_cfg.get("n_channels", 3))
    n_classes = int(model_cfg.get("n_classes", 14))
    bilinear = bool(model_cfg.get("bilinear", False))

    # Import dataset/model from vendor
    import sys

    sys.path.insert(0, str(VENDOR))
    from unet import UNet  # noqa: E402
    from utils.data_loading import BasicDataset  # noqa: E402

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s", device)

    dataset = BasicDataset(str(imgs_dir), str(masks_dir), scale=scale)
    n_val = int(len(dataset) * (val_percent / 100.0))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear).to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ce = nn.CrossEntropyLoss()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = []
    best_miou = -1.0
    best_path = None

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            targets = batch["mask"].to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                logits = model(images)
                loss = ce(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())
            n_batches += 1

        train_loss = 0.0 if n_batches == 0 else running / n_batches
        val_metrics = evaluate_epoch(model, val_loader, device, n_classes, amp)

        ckpt_path = args.save_dir / f"checkpoint_epoch{epoch}.pth"
        state = model.state_dict()
        # Keep mask_values for vendor predict compatibility (optional)
        state["mask_values"] = dataset.mask_values
        torch.save(state, ckpt_path)

        logging.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  mIoU=%.2f  FWIoU=%.2f  Dice=%.2f",
            epoch,
            epochs,
            train_loss,
            val_metrics["val_loss"],
            val_metrics["miou"],
            val_metrics["fwiou"],
            val_metrics["dice_score"],
        )

        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics, "checkpoint": str(ckpt_path)}
        metrics_log.append(row)

        if val_metrics["miou"] > best_miou:
            best_miou = float(val_metrics["miou"])
            best_path = ckpt_path

    out_json = MODULE_ROOT / "results" / "uavscenes_training_log.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics_log, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote training log: %s", out_json)
    if best_path:
        logging.info("Best checkpoint by val mIoU: %s (%.2f)", best_path, best_miou)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

