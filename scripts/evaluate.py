#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fs_jump3d.evaluation import (
    TestVideoDataset,
    build_report,
    predict_test_videos,
    verify_best_checkpoint,
    write_evaluation_artifacts,
)
from fs_jump3d.training import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fixed Milestone 4 checkpoint on the held-out test split.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models/runs/milestone4/best_model.pt")
    parser.add_argument("--training-summary", type=Path, default=PROJECT_ROOT / "data/training/milestone4/summary.json")
    parser.add_argument("--split-csv", type=Path, default=PROJECT_ROOT / "data/splits/dataset_split.csv")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/raw/fs-jump3d")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/evaluation/milestone5")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    model, checkpoint, preprocess = verify_best_checkpoint(args.checkpoint, args.training_summary)
    dataset = TestVideoDataset(args.split_csv, args.data_root, preprocess)
    print(f"Verified checkpoint epoch {checkpoint['epoch']} and {len(dataset)} test videos in 36 groups", flush=True)
    if args.verify_only:
        return
    device = select_device(args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    predictions = predict_test_videos(model, loader, device)
    if len(predictions) != len(dataset) or {row["filepath"] for row in predictions} != {row["filepath"] for row in dataset.rows}:
        raise ValueError("test prediction coverage is incomplete or duplicated")
    report, groups = build_report(predictions, args.checkpoint, device)
    write_evaluation_artifacts(args.output_dir, predictions, groups, report)
    print(f"Video accuracy: {report['video_level']['accuracy']:.4f}")
    print(f"Video macro F1: {report['video_level']['macro']['f1']:.4f}")
    print(f"Group accuracy: {report['group_level']['accuracy']:.4f}")
    print(f"Wrote evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
