from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from fs_jump3d.dataset import TARGET_CLASSES
from fs_jump3d.evaluation import (
    TestVideoDataset,
    aggregate_by_group,
    classification_metrics,
    predict_test_videos,
    read_video_predictions,
    validate_test_split,
    verify_best_checkpoint,
    write_csv,
)
from fs_jump3d.model import CNNBiLSTM, ModelConfig
from fs_jump3d.preprocessing import PreprocessConfig
from fs_jump3d.training import save_checkpoint


def make_split_rows() -> list[dict[str, str]]:
    rows = []
    for split in ("train", "val", "test"):
        for camera in range(1, 13) if split == "test" else (1,):
            rows.append({
                "filepath": f"skater_A/cam_{camera}/Axel_{split}.mp4",
                "group_id": f"A_Axel_{split}", "skater": "A", "camera": str(camera),
                "jump_type": "Axel", "is_target_class": "True", "split": split,
            })
    return rows


def test_test_dataset_selects_only_held_out_rows(tmp_path: Path) -> None:
    rows = make_split_rows()
    index = tmp_path / "split.csv"
    with index.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    dataset = TestVideoDataset(index, tmp_path, PreprocessConfig(), expected_videos=12, expected_groups=1)
    assert len(dataset) == 12
    assert all(row["split"] == "test" for row in dataset.rows)
    assert {row["camera"] for row in dataset.rows} == {str(value) for value in range(1, 13)}
    rows[0]["group_id"] = "A_Axel_test"
    with pytest.raises(ValueError, match="leaks"):
        validate_test_split(rows, expected_videos=12, expected_groups=1)


def test_checkpoint_verification_uses_best_epoch(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pt"
    summary_path = tmp_path / "summary.json"
    model = CNNBiLSTM(ModelConfig(feature_dim=16, lstm_hidden_size=8))
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(checkpoint_path, model, optimizer, 19, 0.78, 0.67, PreprocessConfig(32, 224))
    summary_path.write_text(json.dumps({"best_epoch": 19, "best_val_loss": 0.78}))
    loaded, checkpoint, config = verify_best_checkpoint(checkpoint_path, summary_path)
    assert isinstance(loaded, CNNBiLSTM)
    assert checkpoint["label_mapping"] == list(TARGET_CLASSES)
    assert config == PreprocessConfig(32, 224)
    summary_path.write_text(json.dumps({"best_epoch": 24, "best_val_loss": 0.78}))
    with pytest.raises(ValueError, match="epoch 19"):
        verify_best_checkpoint(checkpoint_path, summary_path)
    with pytest.raises(ValueError, match="epoch 30"):
        verify_best_checkpoint(checkpoint_path, summary_path, expected_epoch=30)
    save_checkpoint(checkpoint_path, model, optimizer, 30, 0.76, 0.70, PreprocessConfig(32, 224))
    summary_path.write_text(json.dumps({"best_epoch": 30, "best_val_loss": 0.76}))
    _, selected, _ = verify_best_checkpoint(checkpoint_path, summary_path, expected_epoch=30)
    assert selected["epoch"] == 30


def test_prediction_shape_and_class_mapping() -> None:
    model = CNNBiLSTM(ModelConfig(feature_dim=16, lstm_hidden_size=8))
    videos = torch.rand(2, 2, 3, 64, 64)
    labels = torch.tensor([0, 5])
    metadata = {"filepath": ["first.mp4", "second.mp4"], "group_id": ["g1", "g2"],
                "skater": ["A", "B"], "camera": ["1", "2"], "jump_type": ["Axel", "Toeloop"]}
    rows = predict_test_videos(model, [(videos, labels, metadata)], torch.device("cpu"))
    assert len(rows) == 2
    assert [row["true_label"] for row in rows] == ["Axel", "Toeloop"]
    assert all(len(row["probabilities"]) == 6 for row in rows)
    assert all(sum(row["probabilities"]) == pytest.approx(1.0) for row in rows)


def test_metrics_and_group_probability_aggregation() -> None:
    def row(true_label: str, predicted_label: str, probabilities: list[float], group_id: str) -> dict:
        return {"true_label": true_label, "predicted_label": predicted_label,
                "probabilities": probabilities, "group_id": group_id, "skater": "A"}

    rows = [row("Axel", "Axel", [0.9, 0.1, 0, 0, 0, 0], "g1") for _ in range(7)]
    rows += [row("Axel", "Flip", [0.2, 0.8, 0, 0, 0, 0], "g1") for _ in range(5)]
    groups = aggregate_by_group(rows)
    assert len(groups) == 1
    assert groups[0]["predicted_label"] == "Axel"
    assert groups[0]["confidence"] == pytest.approx((7 * 0.9 + 5 * 0.2) / 12)
    metrics = classification_metrics([row("Axel", "Axel", [1, 0, 0, 0, 0, 0], "g1"),
                                      row("Flip", "Axel", [1, 0, 0, 0, 0, 0], "g2")])
    assert metrics["accuracy"] == 0.5
    assert len(metrics["confusion_matrix"]) == 6
    assert all(len(line) == 6 for line in metrics["confusion_matrix"])
    assert metrics["per_class"]["Axel"]["support"] == 1


def test_prediction_csv_roundtrip(tmp_path: Path) -> None:
    row = {"filepath": "skater_A/cam_1/Axel_1.mp4", "group_id": "A_Axel_1", "skater": "A",
           "camera": 1, "true_label": "Axel", "predicted_label": "Axel",
           "confidence": 0.7, "correct": True, "probabilities": [0.7, 0.1, 0.1, 0.1, 0, 0]}
    path = tmp_path / "predictions.csv"
    fields = ["filepath", "group_id", "skater", "camera", "true_label", "predicted_label",
              "confidence", "correct"] + [f"prob_{label}" for label in TARGET_CLASSES]
    write_csv(path, [row], fields)
    assert read_video_predictions(path) == [row]
