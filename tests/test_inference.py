from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch

from fs_jump3d.dataset import TARGET_CLASSES
from fs_jump3d.inference import (
    CheckpointError,
    InferenceInputError,
    VideoPredictor,
    validate_video_path,
)
from fs_jump3d.model import CNNBiLSTM
from fs_jump3d.preprocessing import PreprocessConfig, VideoDecodeError
from fs_jump3d.training import save_checkpoint


def write_video(path: Path, frames: int = 3) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (16, 16))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((16, 16, 3), index * 40, dtype=np.uint8))
    writer.release()


@pytest.fixture
def checkpoint(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "best_model.pt"
    model = CNNBiLSTM()
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(path, model, optimizer, 19, 0.78, 0.67, PreprocessConfig(32, 224))
    return path, sha256(path.read_bytes()).hexdigest()


def test_checkpoint_and_single_video_inference(tmp_path: Path, checkpoint, monkeypatch) -> None:
    path, digest = checkpoint
    video = tmp_path / "jump.mp4"
    write_video(video)
    predictor = VideoPredictor.load(path, expected_sha256=digest, device="cpu")
    assert not predictor.model.training
    assert predictor.labels == TARGET_CLASSES
    assert predictor.preprocess == PreprocessConfig(32, 224)

    original_forward = predictor.model.forward
    observations = []

    def checked_forward(batch: torch.Tensor) -> torch.Tensor:
        observations.append((tuple(batch.shape), torch.is_inference_mode_enabled()))
        return original_forward(batch)

    monkeypatch.setattr(predictor.model, "forward", checked_forward)
    first = predictor.predict(video)
    second = predictor.predict(video)
    assert observations == [((1, 32, 3, 224, 224), True)] * 2
    assert first.predicted_label in TARGET_CLASSES
    assert first.predicted_label == max(first.class_probabilities, key=first.class_probabilities.get)
    assert len(first.class_probabilities) == 6
    assert tuple(first.class_probabilities) == TARGET_CLASSES
    assert sum(first.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert 0 <= first.confidence <= 1
    assert first.confidence == first.class_probabilities[first.predicted_label]
    assert first.video_metadata["sampled_frames"] == 32
    assert first.to_dict()["predicted_label"] == first.predicted_label
    assert first.preprocessing_seconds >= 0 and first.model_seconds >= 0
    assert second.class_probabilities == pytest.approx(first.class_probabilities, abs=1e-7)


def test_input_validation_and_decode_errors(tmp_path: Path, checkpoint) -> None:
    path, digest = checkpoint
    predictor = VideoPredictor.load(path, expected_sha256=digest, device="cpu")
    with pytest.raises(InferenceInputError, match="does not exist"):
        predictor.predict(tmp_path / "missing.mp4")
    with pytest.raises(InferenceInputError, match="not a file"):
        predictor.predict(tmp_path)
    text_path = tmp_path / "notes.txt"
    text_path.write_text("not a video")
    with pytest.raises(InferenceInputError, match="MP4"):
        predictor.predict(text_path)
    corrupt_path = tmp_path / "corrupt.mp4"
    corrupt_path.write_text("not an MP4")
    with pytest.raises(VideoDecodeError):
        predictor.predict(corrupt_path)
    empty_video = tmp_path / "empty.mp4"
    write_video(empty_video, frames=0)
    with pytest.raises(VideoDecodeError):
        predictor.predict(empty_video)


def test_checkpoint_errors(tmp_path: Path, checkpoint) -> None:
    path, digest = checkpoint
    with pytest.raises(CheckpointError, match="does not exist"):
        VideoPredictor.load(tmp_path / "missing.pt", expected_sha256=digest)
    with pytest.raises(CheckpointError, match="SHA-256"):
        VideoPredictor.load(path, expected_sha256="0" * 64)
    broken = tmp_path / "broken.pt"
    torch.save({"label_mapping": list(TARGET_CLASSES)}, broken)
    broken_digest = sha256(broken.read_bytes()).hexdigest()
    with pytest.raises(CheckpointError, match="metadata"):
        VideoPredictor.load(broken, expected_sha256=broken_digest)
    incompatible = tmp_path / "incompatible.pt"
    payload = torch.load(path, weights_only=False)
    payload["model_state_dict"]["classifier.4.weight"] = torch.zeros(5, 64)
    torch.save(payload, incompatible)
    incompatible_digest = sha256(incompatible.read_bytes()).hexdigest()
    with pytest.raises(CheckpointError, match="compatible checkpoint"):
        VideoPredictor.load(incompatible, expected_sha256=incompatible_digest)


def test_cli_reports_invalid_path_without_traceback(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/predict_video.py"
    result = subprocess.run([sys.executable, str(script), str(tmp_path / "missing.mp4")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "does not exist" in result.stderr
    assert "Traceback" not in result.stderr
