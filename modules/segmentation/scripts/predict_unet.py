#!/usr/bin/env python3
"""Run Pytorch-UNet prediction; paths resolved relative to modules/segmentation/."""

from __future__ import annotations

import argparse
import logging
import sys
import unittest.mock
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODULE_ROOT = Path(__file__).resolve().parents[1]
VENDOR = MODULE_ROOT / "vendor" / "Pytorch-UNet"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Predict segmentation masks with a trained UNet checkpoint.")
    parser.add_argument("--model", "-m", type=Path, required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--input", "-i", nargs="+", required=True, help="Input image files.")
    parser.add_argument("--output-dir", "-o", type=Path, default=MODULE_ROOT / "results" / "predictions")
    parser.add_argument("--scale", "-s", type=float, default=0.5)
    parser.add_argument("--mask-threshold", "-t", type=float, default=0.5)
    parser.add_argument("--bilinear", action="store_true")
    parser.add_argument("--classes", "-c", type=int, default=2)
    args = parser.parse_args()

    if not (VENDOR / "predict.py").is_file():
        logging.error("Vendor code missing. Run: python modules/segmentation/scripts/setup_vendor.py")
        return 1

    if not args.model.is_file():
        logging.error("Model not found: %s", args.model)
        return 1

    sys.path.insert(0, str(VENDOR))
    sys.modules["wandb"] = unittest.mock.MagicMock()

    from unet import UNet  # noqa: E402
    from utils.data_loading import BasicDataset  # noqa: E402
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s", device)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict)
    net.to(device=device)
    net.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def predict_img(net_mod, full_img: Image.Image, scale_factor: float, out_threshold: float):
        img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
        img = img.unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            output = net_mod(img).cpu()
            output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode="bilinear")
            if net_mod.n_classes > 1:
                mask = output.argmax(dim=1)
            else:
                mask = torch.sigmoid(output) > out_threshold
        return mask[0].long().squeeze().numpy()

    def mask_to_image(mask: np.ndarray, mv):
        if isinstance(mv[0], list):
            out = np.zeros((mask.shape[-2], mask.shape[-1], len(mv[0])), dtype=np.uint8)
        elif mv == [0, 1]:
            out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
        else:
            out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
        if mask.ndim == 3:
            mask = np.argmax(mask, axis=0)
        for i, v in enumerate(mv):
            out[mask == i] = v
        return Image.fromarray(out)

    for fn in args.input:
        p = Path(fn)
        logging.info("Predicting %s", p)
        img = Image.open(p)
        mask = predict_img(net, img, args.scale, args.mask_threshold)
        out_path = args.output_dir / f"{p.stem}_mask.png"
        mask_to_image(mask, mask_values).save(out_path)
        logging.info("Saved %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
