#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fs_jump3d.dataset import TARGET_CLASSES
from fs_jump3d.evaluation import classification_metrics, slice_metrics
from fs_jump3d.preprocessing import PreprocessConfig
from fs_jump3d.training import load_checkpoint, select_device
from fs_jump3d.training_data import SplitVideoDataset
from fs_jump3d.reproducibility import capture_run, seal_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split-csv", type=Path, default=ROOT / "data/splits/dataset_split.csv")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/fs-jump3d")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed/frame_cache")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    provenance_dir = args.output.with_name(args.output.name + ".provenance")
    if args.output.exists() or provenance_dir.exists():
        raise FileExistsError("validation output already exists; choose a new output filename")
    model, metadata = load_checkpoint(args.checkpoint)
    device = select_device(args.device)
    model = model.to(device).eval()
    capture_run(provenance_dir, split_csv=args.split_csv, data_root=args.data_root,
                cache_dir=args.cache_dir, checkpoints=(args.checkpoint,), partitions=("val",),
                config={**{key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
                        **metadata["preprocessing_config"], "actual_device": str(device)})
    dataset = SplitVideoDataset(args.split_csv, args.data_root,
                                "val", PreprocessConfig(**metadata["preprocessing_config"]),
                                cache_dir=args.cache_dir)
    with args.split_csv.open(newline="") as file:
        rows = {row["filepath"]: row for row in csv.DictReader(file) if row["split"] == "val"}
    predictions = []
    with torch.inference_mode():
        for videos, labels, paths in DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2):
            probabilities = torch.softmax(model(videos.to(device)), dim=1).cpu().numpy()
            for truth, path, vector in zip(labels.tolist(), paths, probabilities):
                predicted = int(vector.argmax())
                row = rows[path]
                predictions.append({"filepath": path, "group_id": row["group_id"], "camera": int(row["camera"]),
                                    "skater": row["skater"], "true_label": TARGET_CLASSES[truth],
                                    "predicted_label": TARGET_CLASSES[predicted], "correct": predicted == truth,
                                    "probabilities": vector.tolist()})
    report = {"scope": "validation only", "checkpoint": str(args.checkpoint),
              "selected_epoch": metadata["epoch"], "video_level": classification_metrics(predictions),
              "by_camera": slice_metrics(predictions, "camera")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    seal_outputs(provenance_dir, [args.output])
    print(json.dumps({"accuracy": report["video_level"]["accuracy"],
                      "macro_f1": report["video_level"]["macro"]["f1"]}))


if __name__ == "__main__":
    main()
