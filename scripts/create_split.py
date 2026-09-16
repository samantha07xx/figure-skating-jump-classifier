#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fs_jump3d.split import DEFAULT_SEED, DEFAULT_SPLIT_RATIOS, create_grouped_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the grouped FS-Jump3D train/val/test split.")
    parser.add_argument("--index-path", type=Path, default=PROJECT_ROOT / "data/audit/dataset_index.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/splits")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["train"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["val"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    summary = create_grouped_split(
        index_path=args.index_path.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
        ratios=ratios,
    )
    print(f"Created grouped split in {args.output_dir}")
    print(f"Group counts: {summary['group_counts_by_split']}")
    print(f"Video counts: {summary['video_counts_by_split']}")
    print(f"Checks: {summary['checks']}")


if __name__ == "__main__":
    main()
