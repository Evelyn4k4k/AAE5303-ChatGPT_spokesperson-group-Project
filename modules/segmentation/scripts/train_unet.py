#!/usr/bin/env python3
"""Train U-Net using milesial/Pytorch-UNet with paths rooted under modules/segmentation/."""

from __future__ import annotations

import argparse
import logging
import sys
import unittest.mock
from pathlib import Path

import torch
import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]
VENDOR = MODULE_ROOT / "vendor" / "Pytorch-UNet"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _parse_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train UNet for modules/segmentation (Pytorch-UNet backend).")
    parser.add_argument("--config", type=Path, default=MODULE_ROOT / "configs" / "train_default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--validation-percent", type=float, default=None, help="Percent of data for validation (0-100).")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--classes", type=int, default=None)
    parser.add_argument("--load", type=str, default=None, help="Optional checkpoint .pth to resume from.")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed precision.")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision.")
    parser.set_defaults(amp=None)
    parser.add_argument("--bilinear", action="store_true", help="Use bilinear upsampling in UNet.")
    args = parser.parse_args()

    cfg_path = args.config
    if not cfg_path.is_file():
        logging.error("Config not found: %s", cfg_path)
        return 1

    cfg = _load_yaml(cfg_path)
    epochs = int(args.epochs if args.epochs is not None else cfg.get("epochs", 40))
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 4))
    learning_rate = float(args.learning_rate if args.learning_rate is not None else cfg.get("learning_rate", 1e-4))
    val_percent = float(
        args.validation_percent if args.validation_percent is not None else cfg.get("validation_percent", 10.0)
    )
    scale = float(args.scale if args.scale is not None else cfg.get("scale", 0.5))
    classes = int(args.classes if args.classes is not None else cfg.get("classes", 2))
    bilinear = bool(args.bilinear or cfg.get("bilinear", False))
    load_checkpoint = args.load if args.load is not None else cfg.get("load_checkpoint")

    if args.amp is None:
        amp = _parse_bool(cfg.get("amp", True), True)
    else:
        amp = bool(args.amp)

    if not (VENDOR / "train.py").is_file():
        logging.error("Vendor code missing. Run: python modules/segmentation/scripts/setup_vendor.py")
        return 1

    dir_img = MODULE_ROOT / "datasets" / "imgs"
    dir_mask = MODULE_ROOT / "datasets" / "masks"
    if not dir_img.is_dir() or not any(dir_img.iterdir()):
        logging.error("No images found in %s — see datasets/README.md", dir_img)
        return 1
    if not dir_mask.is_dir() or not any(dir_mask.iterdir()):
        logging.error("No masks found in %s — see datasets/README.md", dir_mask)
        return 1

    # Pytorch-UNet uses wandb; mock it so training works offline / without an account.
    sys.path.insert(0, str(VENDOR))
    sys.modules["wandb"] = unittest.mock.MagicMock()

    import train as vendor_train  # noqa: E402  (after sys.path / wandb mock)

    vendor_train.dir_img = dir_img
    vendor_train.dir_mask = dir_mask
    vendor_train.dir_checkpoint = MODULE_ROOT / "checkpoints"

    from unet import UNet  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s", device)

    model = UNet(n_channels=3, n_classes=classes, bilinear=bilinear)
    model = model.to(memory_format=torch.channels_last)
    logging.info(
        "Network: %s input channels, %s output classes, %s upsampling",
        model.n_channels,
        model.n_classes,
        "bilinear" if bilinear else "transposed conv",
    )

    if load_checkpoint:
        ckpt = Path(str(load_checkpoint))
        if not ckpt.is_file():
            logging.error("Checkpoint not found: %s", ckpt)
            return 1
        state_dict = torch.load(ckpt, map_location=device)
        if "mask_values" in state_dict:
            del state_dict["mask_values"]
        model.load_state_dict(state_dict)
        logging.info("Loaded weights from %s", ckpt)

    model.to(device=device)

    def _run_train() -> None:
        vendor_train.train_model(
            model=model,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_percent=val_percent / 100.0,
            save_checkpoint=True,
            img_scale=scale,
            amp=amp,
        )

    try:
        _run_train()
    except RuntimeError as exc:
        msg = str(exc)
        if "OutOfMemoryError" in type(exc).__name__ or "CUDA out of memory" in msg:
            logging.error("CUDA OOM — retrying with gradient checkpointing (slower). Try smaller batch or --no-amp.")
            torch.cuda.empty_cache()
            model.use_checkpointing()
            _run_train()
        else:
            raise

    logging.info("Training finished. Checkpoints under %s", MODULE_ROOT / "checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
