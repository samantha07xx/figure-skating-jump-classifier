from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from fs_jump3d.dataset import TARGET_CLASSES
from fs_jump3d.model import CNNBiLSTM, ModelConfig
from fs_jump3d.preprocessing import PreprocessConfig
from fs_jump3d.training import build_optimizer, build_scheduler, load_checkpoint, save_checkpoint
from fs_jump3d.training import load_config
from fs_jump3d.training_data import LABEL_TO_INDEX, SplitVideoDataset, augment_clip, make_loaders


def test_model_forward_backward() -> None:
    model = CNNBiLSTM(ModelConfig(feature_dim=16, lstm_hidden_size=8, frame_chunk_size=2))
    videos = torch.rand(2, 3, 3, 64, 64)
    logits = model(videos)
    assert logits.shape == (2, 6)
    loss = torch.nn.CrossEntropyLoss()(logits, torch.tensor([0, 5]))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.frame_encoder.features[1].weight.grad is not None
    assert torch.isfinite(model.frame_encoder.features[1].weight.grad).all()


def test_checkpoint_roundtrip(tmp_path) -> None:
    model = CNNBiLSTM(ModelConfig(feature_dim=16, lstm_hidden_size=8))
    optimizer = torch.optim.Adam(model.parameters())
    inputs = torch.rand(1, 2, 3, 64, 64)
    model.eval()
    expected = model(inputs).detach()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizer, 2, 0.5, 0.75, PreprocessConfig(2, 32))
    loaded, metadata = load_checkpoint(path)
    loaded.eval()
    torch.testing.assert_close(loaded(inputs), expected)
    assert metadata["label_mapping"] == list(TARGET_CLASSES)
    assert metadata["epoch"] == 2


def test_dataset_split_and_mapping(tmp_path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    video = root / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (16, 16))
    for _ in range(2):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()
    index = tmp_path / "split.csv"
    with index.open("w", newline="") as file:
        out = csv.DictWriter(file, fieldnames=["filepath", "jump_type", "split", "is_target_class"])
        out.writeheader()
        for split in ("train", "val", "test"):
            out.writerow({"filepath": "clip.mp4", "jump_type": "Axel", "split": split, "is_target_class": "True"})
    config = PreprocessConfig(2, 16)
    train, val = make_loaders(index, root, config, 1, 0, 42)
    assert len(train.dataset) == len(val.dataset) == 1
    assert all(row["split"] == "train" for row in train.dataset.rows)
    assert all(row["split"] == "val" for row in val.dataset.rows)
    frames, label, path = train.dataset[0]
    assert frames.shape == (2, 3, 16, 16)
    assert label == 0 and path == "clip.mp4"
    assert LABEL_TO_INDEX == {name: i for i, name in enumerate(TARGET_CLASSES)}
    with pytest.raises(ValueError):
        SplitVideoDataset(index, root, "test", config)


def test_training_config_resolves_paths_and_preserves_values(tmp_path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "seed: 42\nframes_per_clip: 32\nimage_size: 224\nbatch_size: 4\n"
        "num_workers: 2\nsplit_csv: data/splits/dataset_split.csv\n"
        "data_root: data/raw/fs-jump3d\ncache_dir: data/processed/frame_cache\n"
        "run_dir: models/runs/milestone4\n"
    )
    config = load_config(config_path, tmp_path)
    assert {key: config[key] for key in ("seed", "frames_per_clip", "image_size", "batch_size", "num_workers")} == {
        "seed": 42, "frames_per_clip": 32, "image_size": 224, "batch_size": 4, "num_workers": 2,
    }
    assert config["split_csv"] == str(tmp_path / "data/splits/dataset_split.csv")


def test_augmentation_is_clip_consistent_and_train_only(tmp_path) -> None:
    frames = np.full((3, 3, 16, 16), 0.5, dtype=np.float32)
    augmented = augment_clip(frames, np.random.default_rng(42))
    assert augmented.shape == frames.shape
    assert augmented.dtype == np.float32
    assert np.allclose(augmented[0], augmented[1])
    assert np.allclose(augmented[1], augmented[2])
    assert 0 <= augmented.min() <= augmented.max() <= 1
    with pytest.raises(ValueError, match="training"):
        SplitVideoDataset(tmp_path / "unused.csv", tmp_path, "val", augment=True)


def test_optimizer_and_scheduler_configuration() -> None:
    model = CNNBiLSTM(ModelConfig(feature_dim=16, lstm_hidden_size=8))
    adam = build_optimizer(model, {"optimizer": "Adam", "learning_rate": 1e-4})
    assert isinstance(adam, torch.optim.Adam)
    assert build_scheduler(adam, {"scheduler": "none"}) is None
    adamw = build_optimizer(model, {"optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4})
    assert isinstance(adamw, torch.optim.AdamW)
    assert adamw.param_groups[0]["weight_decay"] == pytest.approx(1e-4)
    scheduler = build_scheduler(adamw, {"scheduler": "plateau", "scheduler_patience": 2})
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    for _ in range(4):
        scheduler.step(1.0)
    assert adamw.param_groups[0]["lr"] == pytest.approx(5e-5)
    with pytest.raises(ValueError, match="unsupported optimizer"):
        build_optimizer(model, {"optimizer": "invalid", "learning_rate": 1e-4})


def test_phase1_configs_preserve_data_and_model_invariants() -> None:
    root = Path(__file__).resolve().parents[1]
    baseline = yaml.safe_load((root / "configs/train.yaml").read_text())
    names = ("augmentation", "adamw", "plateau", "aug_plateau")
    for name in names:
        config = yaml.safe_load((root / f"configs/improvement_phase1_{name}.yaml").read_text())
        for key in ("seed", "num_classes", "frames_per_clip", "image_size", "batch_size",
                    "epochs", "learning_rate", "early_stopping_patience", "num_workers",
                    "split_csv", "data_root", "cache_dir", "model"):
            assert config[key] == baseline[key]
        assert config["run_dir"] != baseline["run_dir"]
        assert "test" not in config["run_dir"]


def test_validation_loader_remains_unaugmented(tmp_path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    video = root / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (16, 16))
    for _ in range(3):
        writer.write(np.full((16, 16, 3), 128, dtype=np.uint8))
    writer.release()
    index = tmp_path / "split.csv"
    with index.open("w", newline="") as file:
        out = csv.DictWriter(file, fieldnames=["filepath", "jump_type", "split", "is_target_class"])
        out.writeheader()
        for split in ("train", "val"):
            out.writerow({"filepath": "clip.mp4", "jump_type": "Axel", "split": split,
                          "is_target_class": "True"})
    train, val = make_loaders(index, root, PreprocessConfig(2, 16), 1, 0, 42, augment_train=True)
    assert train.dataset.augment is True
    assert val.dataset.augment is False
    torch.testing.assert_close(val.dataset[0][0], val.dataset[0][0])
