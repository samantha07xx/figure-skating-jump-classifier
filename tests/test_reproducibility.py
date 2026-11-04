from __future__ import annotations

import csv
import json
import random

import numpy as np
import pytest
import torch

from fs_jump3d.reproducibility import capture_run, seal_outputs, validate_split_file, verify_manifest


def make_inputs(tmp_path):
    rows = [{"filepath": f"{split}.mp4", "group_id": split, "split": split,
             "jump_type": "Axel"} for split in ("train", "val", "test")]
    split_csv = tmp_path / "split.csv"
    with split_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        (tmp_path / row["filepath"]).write_bytes(b"test video content")
    return split_csv, rows


def test_capture_is_rng_neutral_and_detects_changed_inputs(tmp_path):
    split_csv, _ = make_inputs(tmp_path)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    manifest = capture_run(tmp_path / "record", split_csv=split_csv,
                           data_root=tmp_path, config={})
    assert random.getstate() == before[0]
    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
    assert np.random.get_state()[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])
    assert verify_manifest(manifest) == []
    (tmp_path / "train.mp4").write_bytes(b"replacement content")
    assert any("train.mp4" in error for error in verify_manifest(manifest))
    with pytest.raises(FileExistsError):
        capture_run(tmp_path / "record", split_csv=split_csv, data_root=tmp_path, config={})


def test_cache_and_outputs_are_fingerprinted(tmp_path):
    split_csv, _ = make_inputs(tmp_path)
    cache = tmp_path / "cache" / "2x16"
    cache.mkdir(parents=True)
    np.save(cache / "train.npy", np.zeros((2, 3, 16, 16), dtype=np.uint8))
    manifest = capture_run(tmp_path / "record", split_csv=split_csv, data_root=tmp_path,
                           config={"frames_per_clip": 2, "image_size": 16},
                           cache_dir=cache.parent, partitions=("train", "val"))
    record = json.loads(manifest.read_text())
    assert len(record["existing_caches"]) == 1
    assert len(record["missing_caches_at_start"]) == 1
    assert not any(item["path"].endswith("test.mp4") for item in record["inputs"])
    result = tmp_path / "result.json"
    result.write_text("{}")
    seal_outputs(manifest.parent, [result])
    assert verify_manifest(manifest) == []
    result.write_text('{"modified": true}')
    (cache / "train.npy").write_bytes(b"bad cache")
    errors = verify_manifest(manifest)
    assert any("result.json" in error for error in errors)
    assert any("train.npy" in error for error in errors)


def test_train_val_group_leakage_is_rejected(tmp_path):
    split_csv, rows = make_inputs(tmp_path)
    rows[1]["group_id"] = rows[0]["group_id"]
    with split_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="leaks"):
        validate_split_file(split_csv)


def test_training_record_does_not_change_optimization(tmp_path, monkeypatch):
    """Two tiny real training runs: instrumentation on/off must yield identical weights."""
    import fs_jump3d.training as training
    split_csv, _ = make_inputs(tmp_path)
    videos = torch.linspace(0, 1, 2 * 2 * 3 * 64 * 64).reshape(2, 2, 3, 64, 64)
    batch = (videos, torch.tensor([0, 1]), ["a", "b"])
    monkeypatch.setattr(training, "make_loaders", lambda *a, **k: ([batch], [batch]))
    config = {"seed": 42, "device": "cpu", "frames_per_clip": 2, "image_size": 64,
              "num_classes": 6, "model": {"feature_dim": 8, "lstm_hidden_size": 8},
              "learning_rate": 0.0001, "split_csv": str(split_csv), "data_root": str(tmp_path),
              "batch_size": 2, "num_workers": 0, "epochs": 2, "early_stopping_patience": 5,
              "run_dir": str(tmp_path / "recorded")}
    recorded = training.train(config)
    monkeypatch.setattr(training, "capture_run", lambda *a, **k: None)
    monkeypatch.setattr(training, "seal_outputs", lambda *a, **k: None)
    config["run_dir"] = str(tmp_path / "control")
    control = training.train(config)
    assert recorded["best_val_loss"] == control["best_val_loss"]
    assert recorded["best_val_accuracy"] == control["best_val_accuracy"]
    left = torch.load(tmp_path / "recorded/best_model.pt", weights_only=False)
    right = torch.load(tmp_path / "control/best_model.pt", weights_only=False)
    for key in left["model_state_dict"]:
        assert torch.equal(left["model_state_dict"][key], right["model_state_dict"][key])
