#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fs_jump3d.inference import CheckpointError, InferenceInputError, VideoPredictor, validate_video_path
from fs_jump3d.preprocessing import VideoDecodeError


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify one already-trimmed single-jump MP4.")
    parser.add_argument("video", type=Path, help="path to one trimmed MP4 clip")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default=None)
    args = parser.parse_args()
    try:
        video = validate_video_path(args.video)
        predictor = VideoPredictor.from_config(device_override=args.device)
        result = predictor.predict(video)
    except (CheckpointError, InferenceInputError, VideoDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Prediction: {result.predicted_label}")
    print(f"Confidence: {result.confidence:.2%}")
    print("Class probabilities:")
    for label, probability in result.class_probabilities.items():
        print(f"  {label}: {probability:.2%}")
    print(f"Device: {result.device}")
    print(f"Model: {result.model_id}")
    print(f"Timing: load {predictor.load_seconds:.3f}s, preprocess {result.preprocessing_seconds:.3f}s, "
          f"model {result.model_seconds:.3f}s, total per video {result.total_seconds:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
