#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fs_jump3d.audit import run_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the extracted FS-Jump3D RGB video dataset.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/raw/fs-jump3d")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    if not data_root.exists():
        raise SystemExit(f"Dataset root does not exist: {data_root}")

    summary = run_audit(data_root=data_root, output_dir=output_dir)
    print(f"Indexed {summary['total_real_mp4_files']} real MP4 files")
    print(f"Usable six-class videos: {summary['usable_six_class_mp4_files']}")
    print(f"Physical attempts, six-class: {summary['physical_jump_attempts_six_class']}")
    print(f"Wrote audit artifacts to {output_dir}")


if __name__ == "__main__":
    main()
