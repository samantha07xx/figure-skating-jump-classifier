#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fs_jump3d.training import load_config, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/train.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-dir", type=Path, help="Use a new directory without editing the reference config")
    parser.add_argument("--device", help="Explicit device for reproduction; default preserves config behavior")
    args = parser.parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    if args.run_dir is not None:
        config["run_dir"] = str(args.run_dir.resolve())
    if args.device is not None:
        config["device"] = args.device
    summary = train(config, smoke=args.smoke)
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
