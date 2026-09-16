#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fs_jump3d.preprocessing import (
    PreprocessConfig,
    validate_split_videos,
    write_decode_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OpenCV decoding and preprocessing for split videos.")
    parser.add_argument("--split-csv", type=Path, default=PROJECT_ROOT / "data/splits/dataset_split.csv")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/raw/fs-jump3d")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/preprocessing")
    parser.add_argument("--frames-per-clip", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--inspect-count", type=int, default=3)
    parser.add_argument("--full-decode", action="store_true", help="Decode every frame instead of probing the first frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PreprocessConfig(frames_per_clip=args.frames_per_clip, image_size=args.image_size)
    validation_rows, summary = validate_split_videos(
        split_csv_path=args.split_csv.resolve(),
        data_root=args.data_root.resolve(),
        inspect_count=args.inspect_count,
        config=config,
        full_decode=args.full_decode,
    )
    write_decode_artifacts(validation_rows, summary, args.output_dir.resolve())
    print(f"Validated {summary['total_videos']} videos")
    print(f"Successfully opened: {summary['successfully_opened']}")
    print(f"Successfully decoded: {summary['successfully_decoded']}")
    print(f"Failed videos: {summary['failed_videos']}")
    print(f"Frame counts: min={summary['min_frame_count']} max={summary['max_frame_count']} mean={summary['mean_frame_count']:.2f}")
    print(f"Durations: min={summary['min_duration_seconds']:.4f}s max={summary['max_duration_seconds']:.4f}s mean={summary['mean_duration_seconds']:.4f}s")
    print(f"Wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
