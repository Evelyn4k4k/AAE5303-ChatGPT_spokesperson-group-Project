#!/usr/bin/env python3
"""
End-to-end smoke test: demo data -> short train -> predict -> evaluate -> write results/smoke_metrics.json

Does not overwrite final_candidate/ unless --write-final is set.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]


def _latest_checkpoint(ckpt_dir: Path) -> Path:
    files = list(ckpt_dir.glob("checkpoint_epoch*.pth"))
    if not files:
        raise FileNotFoundError(f"No checkpoint_epoch*.pth under {ckpt_dir}")

    def epoch_key(p: Path) -> int:
        m = re.search(r"checkpoint_epoch(\d+)\.pth$", p.name)
        return int(m.group(1)) if m else -1

    return max(files, key=epoch_key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run segmentation pipeline smoke test.")
    parser.add_argument("--write-final", action="store_true", help="Also copy metrics to final_candidate/metrics_best.json")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch is not installed in this Python environment.")
        print("Activate your venv first, then re-run, e.g.:")
        print("  .\\.venv\\Scripts\\Activate.ps1")
        print("  pip install -r modules\\segmentation\\requirements.txt")
        return 1

    py = sys.executable
    scripts = MODULE_ROOT / "scripts"

    subprocess.check_call([py, str(scripts / "generate_demo_dataset.py"), "--num", "48", "--size", "128"])
    subprocess.check_call(
        [py, str(scripts / "train_unet.py"), "--config", str(MODULE_ROOT / "configs" / "train_smoke.yaml")]
    )

    ckpt = _latest_checkpoint(MODULE_ROOT / "checkpoints")
    imgs = sorted((MODULE_ROOT / "datasets" / "imgs").glob("*.png"))
    if not imgs:
        print("No demo images found.")
        return 1

    pred_cmd = [
        py,
        str(scripts / "predict_unet.py"),
        "--model",
        str(ckpt),
        "--scale",
        "0.5",
        "--classes",
        "2",
        "-i",
        *[str(p) for p in imgs],
        "--output-dir",
        str(MODULE_ROOT / "results" / "predictions"),
    ]
    subprocess.check_call(pred_cmd)

    metrics_path = MODULE_ROOT / "results" / "smoke_metrics.json"
    subprocess.check_call(
        [
            py,
            str(scripts / "evaluate_masks.py"),
            "--pred-dir",
            str(MODULE_ROOT / "results" / "predictions"),
            "--gt-dir",
            str(MODULE_ROOT / "datasets" / "masks"),
            "--num-classes",
            "2",
            "--json-out",
            str(metrics_path),
        ]
    )

    print(f"\nSmoke metrics written to: {metrics_path}")
    print(f"Best checkpoint: {ckpt}")

    if args.write_final:
        import shutil

        final_m = MODULE_ROOT / "final_candidate" / "metrics_best.json"
        shutil.copy2(metrics_path, final_m)
        subprocess.check_call([py, str(scripts / "export_leaderboard_json.py")])
        print(f"Updated {final_m} and final_candidate/ChatGPT_spokesperson_leaderboard.json")

    print("\nNote: smoke numbers are from synthetic data — replace with real AMtown02 labels for submission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
