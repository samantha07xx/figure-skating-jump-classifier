#!/usr/bin/env python3
"""Capture current research inputs or verify an existing manifest, without training."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fs_jump3d.reproducibility import capture_run, verify_manifest
from fs_jump3d.training import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    capture = command.add_parser("capture")
    capture.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--checkpoint", type=Path, action="append", default=[])
    verify = command.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "verify":
        errors = verify_manifest(args.manifest, args.root)
        if errors:
            print("\n".join(errors))
            raise SystemExit(1)
        print("All recorded source, data, cache, snapshot and available result fingerprints match.")
    else:
        config = load_config(args.config, ROOT)
        manifest = capture_run(args.output_dir, split_csv=Path(config["split_csv"]),
                               data_root=Path(config["data_root"]), config=config,
                               cache_dir=Path(config["cache_dir"]) if config.get("cache_dir") else None,
                               checkpoints=tuple(args.checkpoint))
        print(manifest)


if __name__ == "__main__":
    main()
