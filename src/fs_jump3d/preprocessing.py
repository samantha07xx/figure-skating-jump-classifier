from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
import csv
import json

import cv2
import numpy as np


DEFAULT_FRAMES_PER_CLIP = 32
DEFAULT_IMAGE_SIZE = 224


class VideoDecodeError(ValueError):
    """Raised when a video cannot be opened or decoded into frames."""


@dataclass(frozen=True)
class PreprocessConfig:
    frames_per_clip: int = DEFAULT_FRAMES_PER_CLIP
    image_size: int = DEFAULT_IMAGE_SIZE


@dataclass(frozen=True)
class VideoPreprocessingResult:
    frames: np.ndarray
    sampling_indices: list[int]
    decoded_frame_count: int
    fps: float
    duration_seconds: float


def uniform_sample_indices(frame_count: int, frames_per_clip: int = DEFAULT_FRAMES_PER_CLIP) -> list[int]:
    """Return deterministic indices spanning the full decoded clip.

    If a video has fewer decoded frames than requested, every available frame is
    used in order and the final valid frame is repeated until the requested
    length is reached.
    """
    if frames_per_clip <= 0:
        raise ValueError("frames_per_clip must be positive")
    if frame_count <= 0:
        raise VideoDecodeError("video has zero decodable frames")

    if frame_count < frames_per_clip:
        return list(range(frame_count)) + [frame_count - 1] * (frames_per_clip - frame_count)

    return np.linspace(0, frame_count - 1, num=frames_per_clip).round().astype(int).tolist()


def read_video_frames(video_path: Path) -> tuple[list[np.ndarray], float, float]:
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoDecodeError(f"could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise VideoDecodeError(f"video has zero decodable frames: {video_path}")

    duration = len(frames) / fps if fps > 0 else 0.0
    return frames, fps, duration


def preprocess_bgr_frame(frame: np.ndarray, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))


def preprocess_frame_sequence(
    frames: list[np.ndarray],
    config: PreprocessConfig | None = None,
) -> tuple[np.ndarray, list[int]]:
    config = config or PreprocessConfig()
    indices = uniform_sample_indices(len(frames), config.frames_per_clip)
    processed = [preprocess_bgr_frame(frames[index], config.image_size) for index in indices]
    return np.stack(processed, axis=0).astype(np.float32), indices


def preprocess_video(
    video_path: Path,
    config: PreprocessConfig | None = None,
) -> VideoPreprocessingResult:
    config = config or PreprocessConfig()
    frames, fps, duration = read_video_frames(video_path)
    tensor, indices = preprocess_frame_sequence(frames, config)
    return VideoPreprocessingResult(
        frames=tensor,
        sampling_indices=indices,
        decoded_frame_count=len(frames),
        fps=fps,
        duration_seconds=duration,
    )


def preprocess_video_streaming(
    video_path: Path,
    config: PreprocessConfig | None = None,
) -> VideoPreprocessingResult:
    """Decode sequentially while retaining only uniformly selected frames."""
    config = config or PreprocessConfig()
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoDecodeError(f"could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        capture.release()
        return preprocess_video(video_path, config)
    indices = uniform_sample_indices(frame_count, config.frames_per_clip)
    wanted = set(indices)
    selected: dict[int, np.ndarray] = {}
    decoded = 0
    try:
        for frame_index in range(indices[-1] + 1):
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            if frame_index in wanted:
                selected[frame_index] = preprocess_bgr_frame(frame, config.image_size)
    finally:
        capture.release()
    if decoded != indices[-1] + 1:
        return preprocess_video(video_path, config)
    frames = np.stack([selected[index] for index in indices]).astype(np.float32)
    return VideoPreprocessingResult(
        frames=frames,
        sampling_indices=indices,
        decoded_frame_count=decoded,
        fps=fps,
        duration_seconds=decoded / fps if fps > 0 else 0.0,
    )


def validate_video_decode(video_path: Path, full_decode: bool = False) -> dict[str, object]:
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    opened = bool(capture.isOpened())
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) if opened else 0.0
    metadata_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
    frame_count = 0
    error = ""

    if not opened:
        error = "could not open video"
    else:
        try:
            if full_decode:
                while True:
                    ok, _ = capture.read()
                    if not ok:
                        break
                    frame_count += 1
            else:
                ok, _ = capture.read()
                frame_count = metadata_frame_count if ok else 0
        except Exception as exc:  # pragma: no cover - OpenCV exception surface is platform-specific.
            error = f"{type(exc).__name__}: {exc}"
        finally:
            capture.release()

    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    return {
        "opened": opened,
        "decoded": opened and frame_count > 0 and not error,
        "frame_count": frame_count,
        "metadata_frame_count": metadata_frame_count,
        "fps": fps,
        "duration_seconds": duration,
        "error": error,
    }


def summarize_decode_results(rows: list[dict[str, object]]) -> dict[str, object]:
    frame_counts = [int(row["frame_count"]) for row in rows if bool(row["decoded"])]
    durations = [float(row["duration_seconds"]) for row in rows if bool(row["decoded"])]
    unreadable = [row for row in rows if not bool(row["opened"])]
    zero_frame = [row for row in rows if bool(row["opened"]) and int(row["frame_count"]) == 0]
    failed = [row for row in rows if not bool(row["decoded"])]

    return {
        "total_videos": len(rows),
        "successfully_opened": sum(1 for row in rows if bool(row["opened"])),
        "successfully_decoded": sum(1 for row in rows if bool(row["decoded"])),
        "unreadable_videos": len(unreadable),
        "zero_decodable_frame_videos": len(zero_frame),
        "failed_videos": len(failed),
        "min_frame_count": min(frame_counts) if frame_counts else 0,
        "max_frame_count": max(frame_counts) if frame_counts else 0,
        "mean_frame_count": mean(frame_counts) if frame_counts else 0.0,
        "min_duration_seconds": min(durations) if durations else 0.0,
        "max_duration_seconds": max(durations) if durations else 0.0,
        "mean_duration_seconds": mean(durations) if durations else 0.0,
        "unusual_files": [
            {
                "filepath": row["filepath"],
                "error": row["error"],
                "frame_count": row["frame_count"],
            }
            for row in failed
        ],
    }


def read_split_rows(split_csv_path: Path) -> list[dict[str, str]]:
    with Path(split_csv_path).open(newline="") as file:
        return list(csv.DictReader(file))


def validate_split_videos(
    split_csv_path: Path,
    data_root: Path,
    inspect_count: int = 3,
    config: PreprocessConfig | None = None,
    full_decode: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = read_split_rows(split_csv_path)
    config = config or PreprocessConfig()
    validation_rows: list[dict[str, object]] = []
    inspections: list[dict[str, object]] = []

    for row in rows:
        path = Path(data_root) / row["filepath"]
        result = validate_video_decode(path, full_decode=full_decode)
        validation_rows.append({**row, **result})

    for row in rows[:inspect_count]:
        path = Path(data_root) / row["filepath"]
        result = preprocess_video(path, config)
        inspections.append(
            {
                "filepath": row["filepath"],
                "shape": list(result.frames.shape),
                "dtype": str(result.frames.dtype),
                "min_value": float(result.frames.min()),
                "max_value": float(result.frames.max()),
                "sampling_indices": result.sampling_indices,
                "decoded_frame_count": result.decoded_frame_count,
                "fps": result.fps,
                "duration_seconds": result.duration_seconds,
            }
        )

    summary = summarize_decode_results(validation_rows)
    summary["preprocessing_config"] = {
        "frames_per_clip": config.frames_per_clip,
        "image_size": config.image_size,
        "tensor_shape": "[T, C, H, W]",
        "normalization": "[0, 1]",
        "short_video_handling": "use all decoded frames in order, then repeat the final valid frame",
        "augmentation": "none in deterministic Milestone 3 preprocessing",
    }
    summary["validation_mode"] = "full_decode" if full_decode else "open_and_first_frame_decode"
    summary["representative_preprocessing_inspections"] = inspections
    return validation_rows, summary


def write_decode_artifacts(
    validation_rows: list[dict[str, object]],
    summary: dict[str, object],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filepath",
        "split",
        "skater",
        "jump_type",
        "group_id",
        "opened",
        "decoded",
        "frame_count",
        "metadata_frame_count",
        "fps",
        "duration_seconds",
        "error",
    ]
    with (output_dir / "decode_validation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in validation_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    (output_dir / "decode_validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_decode_summary_markdown(summary, output_dir / "decode_validation_summary.md")


def write_decode_summary_markdown(summary: dict[str, object], output_path: Path) -> None:
    inspections = summary["representative_preprocessing_inspections"]
    lines = [
        "# FS-Jump3D Decode Validation Summary",
        "",
        "## Decode Results",
        "",
        f"- Validation mode: {summary['validation_mode']}",
        f"- Total videos: {summary['total_videos']}",
        f"- Successfully opened: {summary['successfully_opened']}",
        f"- Successfully decoded: {summary['successfully_decoded']}",
        f"- Unreadable/corrupt videos: {summary['unreadable_videos']}",
        f"- Zero decodable frame videos: {summary['zero_decodable_frame_videos']}",
        f"- Failed videos: {summary['failed_videos']}",
        "",
        "## Frame Count Statistics",
        "",
        f"- Minimum frame count: {summary['min_frame_count']}",
        f"- Maximum frame count: {summary['max_frame_count']}",
        f"- Mean frame count: {summary['mean_frame_count']:.2f}",
        "",
        "## Duration Statistics",
        "",
        f"- Minimum duration seconds: {summary['min_duration_seconds']:.4f}",
        f"- Maximum duration seconds: {summary['max_duration_seconds']:.4f}",
        f"- Mean duration seconds: {summary['mean_duration_seconds']:.4f}",
        "",
        "## Preprocessing Configuration",
        "",
    ]
    for key, value in summary["preprocessing_config"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Representative Preprocessing Inspections", ""])
    if inspections:
        lines.extend(
            [
                "| Filepath | Shape | Dtype | Min | Max | Decoded frames | First index | Last index |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in inspections:
            indices = item["sampling_indices"]
            lines.append(
                f"| {item['filepath']} | {item['shape']} | {item['dtype']} | "
                f"{item['min_value']:.4f} | {item['max_value']:.4f} | "
                f"{item['decoded_frame_count']} | {indices[0]} | {indices[-1]} |"
            )
    else:
        lines.append("None")
    lines.extend(["", "## Unusual Files", ""])
    if summary["unusual_files"]:
        for item in summary["unusual_files"][:50]:
            lines.append(f"- {item['filepath']}: {item['error']} (frames={item['frame_count']})")
    else:
        lines.append("None")
    output_path.write_text("\n".join(lines) + "\n")
