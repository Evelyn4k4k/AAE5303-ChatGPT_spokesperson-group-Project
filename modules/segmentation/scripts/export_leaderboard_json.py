#!/usr/bin/env python3
"""Build ChatGPT_spokesperson_leaderboard.json from metrics_best.json (UNet leaderboard schema)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "https://github.com/Evelyn4k4k/AAE5303-ChatGPT_spokesperson-group-Project.git"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export UNet-style leaderboard JSON.")
    parser.add_argument("--metrics", type=Path, default=MODULE_ROOT / "final_candidate" / "metrics_best.json")
    parser.add_argument("--out", type=Path, default=MODULE_ROOT / "final_candidate" / "ChatGPT_spokesperson_leaderboard.json")
    parser.add_argument("--group-name", default="ChatGPT_spokesperson")
    parser.add_argument("--repo-url", default=DEFAULT_REPO)
    args = parser.parse_args()

    with args.metrics.open("r", encoding="utf-8") as f:
        m = json.load(f)

    required = ("dice_score", "miou", "fwiou")
    for k in required:
        if k not in m:
            raise SystemExit(f"metrics file missing key: {k}")

    payload = {
        "group_name": args.group_name,
        "project_private_repo_url": args.repo_url,
        "metrics": {k: float(m[k]) for k in required},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
