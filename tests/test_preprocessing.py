from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from fs_jump3d.preprocessing import (
    PreprocessConfig,
    VideoDecodeError,
    preprocess_bgr_frame,
    preprocess_frame_sequence,
    preprocess_video,
    uniform_sample_indices,
)


def solid_bgr_frame(color: tuple[int, int, int], size: int = 8) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def write_synthetic_video(path: Path, frame_count: int, size: tuple[int, int] = (16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        size,
    )
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            value = index % 255
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            frame[:, :, 0] = value
            frame[:, :, 1] = (value * 2) % 255
            frame[:, :, 2] = (value * 3) % 255
            writer.write(frame)
    finally:
        writer.release()


def test_uniform_sampling_is_deterministic_and_spans_clip() -> None:
    first = uniform_sample_indices(64, 32)
    second = uniform_sample_indices(64, 32)

    assert first == second
    assert first[0] == 0
    assert first[-1] == 63
    assert len(first) == 32
    assert all(left <= right for left, right in zip(first, first[1:]))


def test_short_video_sampling_repeats_last_valid_frame() -> None:
    assert uniform_sample_indices(3, 8) == [0, 1, 2, 2, 2, 2, 2, 2]


def test_bgr_to_rgb_conversion_and_normalization() -> None:
    bgr_red = solid_bgr_frame((0, 0, 255))
    processed = preprocess_bgr_frame(bgr_red, image_size=4)

    assert processed.shape == (3, 4, 4)
    assert processed.dtype == np.float32
    assert np.allclose(processed[0], 1.0)
    assert np.allclose(processed[1], 0.0)
    assert np.allclose(processed[2], 0.0)
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_preprocess_frame_sequence_shape_dtype_and_range() -> None:
    frames = [solid_bgr_frame((index, index, index)) for index in range(40)]
    tensor, indices = preprocess_frame_sequence(frames, PreprocessConfig(frames_per_clip=32, image_size=224))

    assert tensor.shape == (32, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    assert indices[0] == 0
    assert indices[-1] == 39


def test_preprocess_frame_sequence_short_video_padding() -> None:
    frames = [
        solid_bgr_frame((0, 0, 0)),
        solid_bgr_frame((10, 10, 10)),
        solid_bgr_frame((20, 20, 20)),
    ]
    tensor, indices = preprocess_frame_sequence(frames, PreprocessConfig(frames_per_clip=5, image_size=8))

    assert tensor.shape == (5, 3, 8, 8)
    assert indices == [0, 1, 2, 2, 2]
    assert np.allclose(tensor[-1], tensor[2])


def test_preprocess_video_returns_pytorch_friendly_tensor(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    write_synthetic_video(video_path, frame_count=36)

    result = preprocess_video(video_path, PreprocessConfig(frames_per_clip=32, image_size=224))

    assert result.frames.shape == (32, 3, 224, 224)
    assert result.frames.dtype == np.float32
    assert result.frames.min() >= 0.0
    assert result.frames.max() <= 1.0
    assert result.sampling_indices[0] == 0
    assert result.sampling_indices[-1] == result.decoded_frame_count - 1


def test_invalid_video_handling_is_explicit(tmp_path: Path) -> None:
    invalid_path = tmp_path / "not_a_video.mp4"
    invalid_path.write_text("not a video")

    with pytest.raises(VideoDecodeError, match="could not open video|zero decodable"):
        preprocess_video(invalid_path)
