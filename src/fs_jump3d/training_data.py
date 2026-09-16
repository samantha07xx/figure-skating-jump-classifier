from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fs_jump3d.dataset import TARGET_CLASSES
from fs_jump3d.preprocessing import PreprocessConfig, preprocess_video_streaming


LABEL_TO_INDEX = {label: index for index, label in enumerate(TARGET_CLASSES)}


class SplitVideoDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        data_root: Path,
        split: str,
        preprocess_config: PreprocessConfig | None = None,
        max_samples: int | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError("training datasets permit only train or val")
        self.data_root = Path(data_root)
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        with Path(split_csv).open(newline="") as file:
            rows = list(csv.DictReader(file))
        if any(row["split"] not in ("train", "val", "test") for row in rows):
            raise ValueError("unassigned or unexpected split in index")
        if any(row["jump_type"] not in LABEL_TO_INDEX or row["is_target_class"] != "True" for row in rows):
            raise ValueError("split index contains non-target labels")
        self.rows = [row for row in rows if row["split"] == split]
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"empty {split} dataset")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows[index]
        video_path = self.data_root / row["filepath"]
        if self.cache_dir is None:
            frames = preprocess_video_streaming(video_path, self.preprocess_config).frames
        else:
            cache_path = self.cache_dir / f"{self.preprocess_config.frames_per_clip}x{self.preprocess_config.image_size}" / Path(row["filepath"]).with_suffix(".npy")
            if cache_path.exists():
                frames = np.load(cache_path).astype(np.float32) / 255.0
            else:
                frames = preprocess_video_streaming(video_path, self.preprocess_config).frames
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp.npy")
                np.save(temporary, np.rint(frames * 255.0).astype(np.uint8))
                temporary.replace(cache_path)
        return torch.from_numpy(frames), LABEL_TO_INDEX[row["jump_type"]], row["filepath"]


def make_loaders(
    split_csv: Path,
    data_root: Path,
    preprocess_config: PreprocessConfig,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_samples: int | None = None,
    cache_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    train = SplitVideoDataset(split_csv, data_root, "train", preprocess_config, max_samples, cache_dir)
    val = SplitVideoDataset(split_csv, data_root, "val", preprocess_config, max_samples, cache_dir)
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
