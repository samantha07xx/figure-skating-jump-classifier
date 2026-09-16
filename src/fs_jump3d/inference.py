from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import pickle

import numpy as np
import torch
import yaml

from fs_jump3d.dataset import TARGET_CLASSES
from fs_jump3d.model import ModelConfig
from fs_jump3d.preprocessing import PreprocessConfig, VideoDecodeError, preprocess_video_streaming
from fs_jump3d.training import load_checkpoint, select_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/inference.yaml"


class InferenceInputError(ValueError):
    """Raised when the requested input is not an MP4 file."""


class CheckpointError(ValueError):
    """Raised when the fixed checkpoint is missing or incompatible."""


@dataclass(frozen=True)
class InferenceResult:
    predicted_label: str
    confidence: float
    class_probabilities: dict[str, float]
    model_id: str
    device: str
    video_metadata: dict[str, int | float]
    preprocessing_seconds: float
    model_seconds: float
    total_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_video_path(path: Path | str) -> Path:
    video_path = Path(path).expanduser()
    if not video_path.exists():
        raise InferenceInputError(f"video file does not exist: {video_path}")
    if not video_path.is_file():
        raise InferenceInputError(f"video path is not a file: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        raise InferenceInputError("input must be an MP4 file")
    return video_path


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class VideoPredictor:
    def __init__(self, model: torch.nn.Module, labels: tuple[str, ...],
                 preprocess: PreprocessConfig, device: torch.device, model_id: str,
                 load_seconds: float) -> None:
        self.model = model.eval()
        self.labels = labels
        self.preprocess = preprocess
        self.device = device
        self.model_id = model_id
        self.load_seconds = load_seconds

    @classmethod
    def from_config(cls, config_path: Path = DEFAULT_CONFIG,
                    device_override: str | None = None) -> VideoPredictor:
        config_path = Path(config_path)
        try:
            config = yaml.safe_load(config_path.read_text())
            checkpoint = PROJECT_ROOT / config["checkpoint"]
            digest = config["expected_sha256"]
            epoch = int(config["expected_epoch"])
            device = device_override or config["device"]
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            raise CheckpointError(f"invalid inference configuration: {exc}") from exc
        return cls.load(checkpoint, expected_sha256=digest, expected_epoch=epoch, device=device)

    @classmethod
    def load(cls, checkpoint_path: Path, *, expected_sha256: str,
             expected_epoch: int = 19, device: str = "auto") -> VideoPredictor:
        start = perf_counter()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise CheckpointError(f"checkpoint does not exist: {checkpoint_path}")
        try:
            digest = sha256(checkpoint_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise CheckpointError(f"cannot read checkpoint: {exc}") from exc
        if digest != expected_sha256:
            raise CheckpointError("checkpoint SHA-256 does not match the fixed Milestone 4 model")
        try:
            model, metadata = load_checkpoint(checkpoint_path, "cpu")
            required = ("epoch", "label_mapping", "model_config", "preprocessing_config", "model_state_dict")
            missing = [key for key in required if key not in metadata]
            if missing:
                raise CheckpointError(f"checkpoint is missing metadata: {', '.join(missing)}")
            labels = tuple(metadata["label_mapping"])
            preprocess = PreprocessConfig(**metadata["preprocessing_config"])
            if metadata["epoch"] != expected_epoch:
                raise CheckpointError(f"expected checkpoint epoch {expected_epoch}, got {metadata['epoch']}")
            if labels != TARGET_CLASSES:
                raise CheckpointError("checkpoint label mapping is invalid")
            if model.config != ModelConfig() or preprocess != PreprocessConfig(32, 224):
                raise CheckpointError("checkpoint model or preprocessing configuration is incompatible")
        except CheckpointError:
            raise
        except KeyError as exc:
            raise CheckpointError(f"checkpoint is missing required metadata: {exc.args[0]}") from exc
        except (TypeError, ValueError, RuntimeError, OSError, EOFError, pickle.UnpicklingError) as exc:
            raise CheckpointError(f"cannot load compatible checkpoint: {exc}") from exc
        chosen_device = torch.device("cpu")
        try:
            chosen_device = select_device(device)
            model = model.to(chosen_device).eval()
        except RuntimeError as exc:
            if device == "auto" and chosen_device.type == "mps":
                chosen_device = torch.device("cpu")
                model = model.to(chosen_device).eval()
            else:
                raise CheckpointError(f"cannot use requested device: {exc}") from exc
        return cls(model, labels, preprocess, chosen_device,
                   f"epoch-{expected_epoch}-{digest[:12]}", perf_counter() - start)

    def predict(self, video_path: Path | str) -> InferenceResult:
        start = perf_counter()
        path = validate_video_path(video_path)
        preprocessing_start = perf_counter()
        video = preprocess_video_streaming(path, self.preprocess)
        expected_shape = (self.preprocess.frames_per_clip, 3,
                          self.preprocess.image_size, self.preprocess.image_size)
        if video.frames.shape != expected_shape or video.frames.dtype != np.float32:
            raise VideoDecodeError(f"unexpected preprocessed tensor shape or dtype: {video.frames.shape}")
        preprocessing_seconds = perf_counter() - preprocessing_start

        model_start = perf_counter()
        try:
            with torch.inference_mode():
                batch = torch.from_numpy(video.frames).unsqueeze(0).to(self.device)
                logits = self.model(batch)
                if logits.shape != (1, len(self.labels)) or not torch.isfinite(logits).all():
                    raise CheckpointError("model produced invalid logits")
                probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
                _synchronize(self.device)
        except (RuntimeError, NotImplementedError) as exc:
            if self.device.type != "mps":
                raise CheckpointError(f"model inference failed: {exc}") from exc
            try:
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device).eval()
                with torch.inference_mode():
                    logits = self.model(torch.from_numpy(video.frames).unsqueeze(0))
                    if logits.shape != (1, len(self.labels)) or not torch.isfinite(logits).all():
                        raise CheckpointError("model produced invalid logits after CPU fallback")
                    probabilities = torch.softmax(logits, dim=1)[0].tolist()
            except (RuntimeError, NotImplementedError) as fallback_exc:
                raise CheckpointError(f"model inference failed on MPS and CPU: {fallback_exc}") from fallback_exc
        model_seconds = perf_counter() - model_start
        if (len(probabilities) != len(self.labels)
                or not all(np.isfinite(value) and 0 <= value <= 1 for value in probabilities)
                or abs(sum(probabilities) - 1.0) > 1e-5):
            raise CheckpointError("model produced invalid class probabilities")
        predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return InferenceResult(
            predicted_label=self.labels[predicted_index],
            confidence=probabilities[predicted_index],
            class_probabilities=dict(zip(self.labels, probabilities)),
            model_id=self.model_id,
            device=str(self.device),
            video_metadata={
                "decoded_frames": video.decoded_frame_count,
                "sampled_frames": len(video.sampling_indices),
                "fps": video.fps,
                "duration_seconds": video.duration_seconds,
            },
            preprocessing_seconds=preprocessing_seconds,
            model_seconds=model_seconds,
            total_seconds=perf_counter() - start,
        )
