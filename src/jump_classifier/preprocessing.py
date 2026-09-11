from pathlib import Path

import cv2
import numpy as np


DEFAULT_NUM_FRAMES = 30
DEFAULT_IMAGE_SIZE = (160, 160)


def sample_video_frames(
    video_path: str | Path,
    num_frames: int = DEFAULT_NUM_FRAMES,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Read a video and return a fixed-length normalized RGB frame sequence."""
    video_path = str(video_path)
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise ValueError(f"Video has no readable frames: {video_path}")

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []

    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, image_size)
        frame = frame.astype("float32") / 255.0
        frames.append(frame)

    capture.release()

    if not frames:
        raise ValueError(f"No frames could be sampled from: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return np.asarray(frames[:num_frames], dtype="float32")
