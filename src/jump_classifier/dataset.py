from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from .labels import CLASS_NAMES, LABEL_TO_INDEX, normalize_label_name
from .preprocessing import DEFAULT_IMAGE_SIZE, DEFAULT_NUM_FRAMES, sample_video_frames

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def find_labeled_videos(data_dir: str | Path) -> list[tuple[Path, int]]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    samples = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        label = normalize_label_name(class_dir.name)
        label_index = LABEL_TO_INDEX[label]

        for video_path in sorted(class_dir.rglob("*")):
            if video_path.suffix.lower() in VIDEO_EXTENSIONS:
                samples.append((video_path, label_index))

    if not samples:
        raise ValueError(f"No video files found in {data_dir}")

    return samples


def load_dataset(
    data_dir: str | Path,
    num_frames: int = DEFAULT_NUM_FRAMES,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    samples = find_labeled_videos(data_dir)
    x_values = []
    y_values = []
    paths = []

    for video_path, label_index in samples:
        x_values.append(sample_video_frames(video_path, num_frames, image_size))
        y_values.append(label_index)
        paths.append(video_path)

    x = np.asarray(x_values, dtype="float32")
    y = to_categorical(np.asarray(y_values), num_classes=len(CLASS_NAMES))
    return x, y, paths


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    labels = np.argmax(y, axis=1)

    stratify = labels if _can_stratify(labels) else None
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    train_val_labels = np.argmax(y_train_val, axis=1)
    adjusted_val_size = val_size / (1.0 - test_size)
    stratify_train_val = train_val_labels if _can_stratify(train_val_labels) else None

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=stratify_train_val,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def _can_stratify(labels: np.ndarray) -> bool:
    unique, counts = np.unique(labels, return_counts=True)
    return len(unique) > 1 and np.all(counts >= 2)
