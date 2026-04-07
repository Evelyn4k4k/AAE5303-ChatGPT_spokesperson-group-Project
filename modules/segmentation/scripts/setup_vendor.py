#!/usr/bin/env python3
"""Clone milesial/Pytorch-UNet into modules/segmentation/vendor/Pytorch-UNet."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = MODULE_ROOT / "vendor" / "Pytorch-UNet"
REPO = "https://github.com/milesial/Pytorch-UNet.git"


def main() -> int:
    if VENDOR_DIR.joinpath("train.py").is_file():
        print(f"Already present: {VENDOR_DIR}")
        return 0

    MODULE_ROOT.joinpath("vendor").mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", REPO, str(VENDOR_DIR)]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
