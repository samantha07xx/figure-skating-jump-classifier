from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset

from fs_jump3d.dataset import TARGET_CLASSES
from fs_jump3d.preprocessing import PreprocessConfig, preprocess_video_streaming
from fs_jump3d.training import load_checkpoint
from fs_jump3d.training_data import LABEL_TO_INDEX


class TestVideoDataset(Dataset):
    __test__ = False

    def __init__(self, split_csv: Path, data_root: Path, config: PreprocessConfig,
                 expected_videos: int = 432, expected_groups: int = 36) -> None:
        self.data_root = Path(data_root)
        self.config = config
        with Path(split_csv).open(newline="") as file:
            all_rows = list(csv.DictReader(file))
        validate_test_split(all_rows, expected_videos=expected_videos, expected_groups=expected_groups)
        self.rows = [row for row in all_rows if row["split"] == "test"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, str]]:
        row = self.rows[index]
        frames = preprocess_video_streaming(self.data_root / row["filepath"], self.config).frames
        metadata = {key: row[key] for key in ("filepath", "group_id", "skater", "camera", "jump_type")}
        return torch.from_numpy(frames), LABEL_TO_INDEX[row["jump_type"]], metadata


def validate_test_split(
    rows: list[dict[str, str]],
    expected_videos: int = 432,
    expected_groups: int = 36,
    views_per_group: int = 12,
) -> None:
    if not rows or any(row["split"] not in ("train", "val", "test") for row in rows):
        raise ValueError("invalid or empty split index")
    test_rows = [row for row in rows if row["split"] == "test"]
    if len(test_rows) != expected_videos:
        raise ValueError(f"expected {expected_videos} test videos, got {len(test_rows)}")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in test_rows:
        if row["jump_type"] not in TARGET_CLASSES or row["is_target_class"] != "True":
            raise ValueError("test split contains a non-target label")
        groups[row["group_id"]].append(row)
    if len(groups) != expected_groups:
        raise ValueError(f"expected {expected_groups} test groups, got {len(groups)}")
    cameras = set(range(1, views_per_group + 1))
    for group_id, group in groups.items():
        if len(group) != views_per_group or {int(row["camera"]) for row in group} != cameras:
            raise ValueError(f"incomplete camera views for {group_id}")
        if len({row["jump_type"] for row in group}) != 1:
            raise ValueError(f"inconsistent labels for {group_id}")
    test_paths = [row["filepath"] for row in test_rows]
    if len(test_paths) != len(set(test_paths)):
        raise ValueError("duplicate test filepath")
    other_rows = [row for row in rows if row["split"] != "test"]
    if set(test_paths) & {row["filepath"] for row in other_rows}:
        raise ValueError("test filepath appears in train/validation")
    if set(groups) & {row["group_id"] for row in other_rows}:
        raise ValueError("physical-attempt group leaks across splits")
    if expected_videos == 432 and Counter(row["jump_type"] for row in test_rows) != {
        label: 72 for label in TARGET_CLASSES
    }:
        raise ValueError("unexpected test class distribution")


def verify_best_checkpoint(checkpoint_path: Path, training_summary_path: Path,
                           expected_epoch: int = 19) -> tuple[torch.nn.Module, dict, PreprocessConfig]:
    model, checkpoint = load_checkpoint(checkpoint_path, "cpu")
    summary = json.loads(Path(training_summary_path).read_text())
    if checkpoint["epoch"] != summary["best_epoch"] or checkpoint["epoch"] != expected_epoch:
        raise ValueError(f"checkpoint is not the validation-selected epoch {expected_epoch} model")
    if not np.isclose(checkpoint["val_loss"], summary["best_val_loss"]):
        raise ValueError("checkpoint validation loss differs from the training record")
    if checkpoint["label_mapping"] != list(TARGET_CLASSES):
        raise ValueError("checkpoint class mapping differs from canonical mapping")
    config = PreprocessConfig(**checkpoint["preprocessing_config"])
    if config != PreprocessConfig(32, 224):
        raise ValueError("checkpoint preprocessing differs from Milestone 3")
    if model.config.num_classes != len(TARGET_CLASSES):
        raise ValueError("checkpoint output class count is incorrect")
    return model, checkpoint, config


def predict_test_videos(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> list[dict[str, object]]:
    model = model.to(device)
    model.eval()
    predictions: list[dict[str, object]] = []
    with torch.inference_mode():
        for videos, labels, metadata in loader:
            logits = model(videos.to(device))
            if logits.shape != (len(labels), len(TARGET_CLASSES)):
                raise ValueError("unexpected model output shape")
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            for index, vector in enumerate(probabilities):
                predicted = int(vector.argmax())
                true_index = int(labels[index])
                predictions.append({
                    "filepath": metadata["filepath"][index],
                    "group_id": metadata["group_id"][index],
                    "skater": metadata["skater"][index],
                    "camera": int(metadata["camera"][index]),
                    "true_label": TARGET_CLASSES[true_index],
                    "predicted_label": TARGET_CLASSES[predicted],
                    "confidence": float(vector[predicted]),
                    "correct": predicted == true_index,
                    "probabilities": [float(value) for value in vector],
                })
    return predictions


def classification_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    truth = [LABEL_TO_INDEX[row["true_label"]] for row in rows]
    predicted = [LABEL_TO_INDEX[row["predicted_label"]] for row in rows]
    labels = list(range(len(TARGET_CLASSES)))
    macro = precision_recall_fscore_support(truth, predicted, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(truth, predicted, labels=labels, average="weighted", zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=labels)
    report = classification_report(truth, predicted, labels=labels, target_names=TARGET_CLASSES,
                                   output_dict=True, zero_division=0)
    return {
        "sample_count": len(rows),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro": {name: float(value) for name, value in zip(("precision", "recall", "f1"), macro[:3])},
        "weighted": {name: float(value) for name, value in zip(("precision", "recall", "f1"), weighted[:3])},
        "per_class": {label: {metric: float(report[label][metric]) for metric in ("precision", "recall", "f1-score", "support")}
                      for label in TARGET_CLASSES},
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(truth, predicted, labels=labels,
                                                       target_names=TARGET_CLASSES, zero_division=0),
    }


def aggregate_by_group(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    aggregated = []
    for group_id, views in sorted(groups.items()):
        if len(views) != 12 or len({view["true_label"] for view in views}) != 1:
            raise ValueError(f"invalid physical-attempt group: {group_id}")
        mean_probability = np.mean([view["probabilities"] for view in views], axis=0)
        predicted_index = int(mean_probability.argmax())
        aggregated.append({
            "group_id": group_id,
            "skater": views[0]["skater"],
            "true_label": views[0]["true_label"],
            "predicted_label": TARGET_CLASSES[predicted_index],
            "confidence": float(mean_probability[predicted_index]),
            "correct": TARGET_CLASSES[predicted_index] == views[0]["true_label"],
            "probabilities": mean_probability.tolist(),
            "view_count": len(views),
        })
    return aggregated


def slice_metrics(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, float | int]]:
    values = sorted({row[key] for row in rows})
    return {str(value): {"videos": len(subset), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro"]["f1"]}
            for value in values
            for subset in [[row for row in rows if row[key] == value]]
            for metrics in [classification_metrics(subset)]}


def error_analysis(rows: list[dict[str, object]], metrics: dict[str, object]) -> dict[str, object]:
    errors = [row for row in rows if not row["correct"]]
    correct = [row for row in rows if row["correct"]]
    pairs = Counter((row["true_label"], row["predicted_label"]) for row in errors)
    per_class = metrics["per_class"]
    ranked = sorted(TARGET_CLASSES, key=lambda label: (per_class[label]["f1-score"], label))
    high_confidence = sorted((row for row in errors if row["confidence"] >= 0.8),
                             key=lambda row: (-row["confidence"], row["filepath"]))
    return {
        "error_count": len(errors),
        "most_confused_pairs": [{"true_label": truth, "predicted_label": predicted, "count": count}
                                for (truth, predicted), count in pairs.most_common()],
        "strongest_class_by_f1": ranked[-1],
        "weakest_class_by_f1": ranked[0],
        "correct_confidence_mean": float(np.mean([row["confidence"] for row in correct])) if correct else None,
        "correct_confidence_median": float(np.median([row["confidence"] for row in correct])) if correct else None,
        "incorrect_confidence_mean": float(np.mean([row["confidence"] for row in errors])) if errors else None,
        "incorrect_confidence_median": float(np.median([row["confidence"] for row in errors])) if errors else None,
        "high_confidence_error_threshold": 0.8,
        "high_confidence_error_count": len(high_confidence),
        "high_confidence_errors": [{key: row[key] for key in ("filepath", "group_id", "true_label", "predicted_label", "confidence")}
                                   for row in high_confidence],
    }


def build_report(predictions: list[dict[str, object]], checkpoint_path: Path, device: torch.device) -> tuple[dict[str, object], list[dict[str, object]]]:
    groups = aggregate_by_group(predictions)
    video_metrics = classification_metrics(predictions)
    checkpoint_path = Path(checkpoint_path).resolve()
    project_root = Path(__file__).resolve().parents[2]
    try:
        display_checkpoint = checkpoint_path.relative_to(project_root).as_posix()
    except ValueError:
        display_checkpoint = str(checkpoint_path)
    report = {
        "checkpoint": display_checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "device": str(device),
        "labels": list(TARGET_CLASSES),
        "group_aggregation": "arithmetic mean of the 12 camera-view class-probability vectors, followed by argmax",
        "video_level": video_metrics,
        "group_level": classification_metrics(groups),
        "by_camera": slice_metrics(predictions, "camera"),
        "by_skater": slice_metrics(predictions, "skater"),
        "errors": error_analysis(predictions, video_metrics),
    }
    return report, groups


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            record = dict(row)
            probabilities = record.pop("probabilities")
            record.update({f"prob_{label}": value for label, value in zip(TARGET_CLASSES, probabilities)})
            writer.writerow({key: record[key] for key in fieldnames})


def read_video_predictions(path: Path) -> list[dict[str, object]]:
    with Path(path).open(newline="") as file:
        rows = list(csv.DictReader(file))
    predictions = []
    for row in rows:
        predictions.append({
            "filepath": row["filepath"],
            "group_id": row["group_id"],
            "skater": row["skater"],
            "camera": int(row["camera"]),
            "true_label": row["true_label"],
            "predicted_label": row["predicted_label"],
            "confidence": float(row["confidence"]),
            "correct": row["correct"] == "True",
            "probabilities": [float(row[f"prob_{label}"]) for label in TARGET_CLASSES],
        })
    return predictions


def write_evaluation_artifacts(output_dir: Path, predictions: list[dict[str, object]],
                               groups: list[dict[str, object]], report: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    probability_columns = [f"prob_{label}" for label in TARGET_CLASSES]
    write_csv(output_dir / "video_predictions.csv", predictions,
              ["filepath", "group_id", "skater", "camera", "true_label", "predicted_label", "confidence", "correct"] + probability_columns)
    write_csv(output_dir / "group_predictions.csv", groups,
              ["group_id", "skater", "true_label", "predicted_label", "confidence", "correct", "view_count"] + probability_columns)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in ("video_level", "group_level"):
        matrix = report[name]["confusion_matrix"]
        with (output_dir / f"{name}_confusion_matrix.csv").open("w", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["true/predicted", *TARGET_CLASSES])
            writer.writerows([[label, *row] for label, row in zip(TARGET_CLASSES, matrix)])
    write_markdown_report(output_dir / "evaluation_summary.md", report)
    write_plots(output_dir, report)


def write_markdown_report(path: Path, report: dict[str, object]) -> None:
    video = report["video_level"]
    group = report["group_level"]
    errors = report["errors"]
    lines = [
        "# Milestone 5 Held-Out Evaluation",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Checkpoint SHA-256: `{report['checkpoint_sha256']}`",
        f"- Device: `{report['device']}`",
        "- Scope: grouped held-out FS-Jump3D test split; no retraining or test-driven model selection",
        "",
        "## Video-Level Performance",
        "",
        f"{video['sample_count']} individual camera-view videos; accuracy **{video['accuracy']:.2%}**, "
        f"macro F1 **{video['macro']['f1']:.3f}**.",
        "",
        "| Average | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for average in ("macro", "weighted"):
        values = video[average]
        lines.append(f"| {average.title()} | {values['precision']:.3f} | {values['recall']:.3f} | {values['f1']:.3f} |")
    lines.extend(["", "| Class | Precision | Recall | F1 | Support |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for label, values in video["per_class"].items():
        lines.append(f"| {label} | {values['precision']:.3f} | {values['recall']:.3f} | "
                     f"{values['f1-score']:.3f} | {int(values['support'])} |")
    lines.extend(["", "## Physical-Attempt Performance", "",
                  f"{group['sample_count']} attempts, each with 12 views. Aggregation: {report['group_aggregation']}.",
                  f"Group accuracy **{group['accuracy']:.2%}**; macro precision **{group['macro']['precision']:.3f}**, "
                  f"recall **{group['macro']['recall']:.3f}**, F1 **{group['macro']['f1']:.3f}**.",
                  "", "## Camera Performance", "",
                  "| Camera | Videos | Accuracy | Macro F1 |", "| ---: | ---: | ---: | ---: |"])
    for camera, values in report["by_camera"].items():
        lines.append(f"| {camera} | {values['videos']} | {values['accuracy']:.2%} | {values['macro_f1']:.3f} |")
    lines.extend(["", "## Skater Performance", "",
                  "| Skater | Videos | Accuracy | Macro F1 |", "| --- | ---: | ---: | ---: |"])
    for skater, values in report["by_skater"].items():
        lines.append(f"| {skater} | {values['videos']} | {values['accuracy']:.2%} | {values['macro_f1']:.3f} |")
    lines.extend(["", "Each skater has all six classes, but their class proportions differ; these descriptive scores are not controlled skater comparisons."])
    lines.extend(["", "## Errors And Confidence", "",
                  f"- Incorrect videos: {errors['error_count']}",
                  f"- Strongest class by F1: {errors['strongest_class_by_f1']}",
                  f"- Weakest class by F1: {errors['weakest_class_by_f1']}",
                  f"- Mean confidence, correct: {errors['correct_confidence_mean']:.3f}",
                  f"- Mean confidence, incorrect: {errors['incorrect_confidence_mean']:.3f}",
                  f"- High-confidence errors (confidence >= 0.8): {errors['high_confidence_error_count']}",
                  "", "| True | Predicted | Count |", "| --- | --- | ---: |"])
    for pair in errors["most_confused_pairs"][:10]:
        lines.append(f"| {pair['true_label']} | {pair['predicted_label']} | {pair['count']} |")
    lines.extend(["", "### High-Confidence Mistakes", "",
                  "| Video | True | Predicted | Confidence |", "| --- | --- | --- | ---: |"])
    for mistake in errors["high_confidence_errors"][:5]:
        lines.append(f"| {mistake['filepath']} | {mistake['true_label']} | "
                     f"{mistake['predicted_label']} | {mistake['confidence']:.3f} |")
    if not errors["high_confidence_errors"]:
        lines.append("| None | - | - | - |")
    lines.extend(["", "## Limitations", "",
                  "FS-Jump3D is controlled, multi-camera footage from four skaters. All four skaters appear in the established train/validation/test split, so this is not unseen-skater evaluation. These results do not establish generalization to broadcast, competition, phone, or internet videos, or to substantially different recording conditions.",
                  "", "The confusion-matrix CSV/PNG files, per-video predictions, and per-group predictions contain the complete results. Observed confusion and viewpoint differences are descriptive; no biomechanical cause is inferred.", ""])
    path.write_text("\n".join(lines))


def write_plots(output_dir: Path, report: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(report["video_level"]["confusion_matrix"], dtype=float)
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                           out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) != 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, values, title, format_string in (
        (axes[0], matrix, "Counts", ".0f"),
        (axes[1], normalized, "Row-normalized", ".2f"),
    ):
        axis.imshow(values, cmap="Blues", vmin=0, vmax=float(values.max()))
        axis.set_title(title)
        axis.set_xticks(range(6), TARGET_CLASSES, rotation=45, ha="right")
        axis.set_yticks(range(6), TARGET_CLASSES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for row in range(6):
            for column in range(6):
                axis.text(column, row, format(values[row, column], format_string),
                          ha="center", va="center", fontsize=8,
                          color="white" if values[row, column] > values.max() / 2 else "black")
    fig.savefig(output_dir / "video_confusion_matrix.png", dpi=140)
    plt.close(fig)

    metrics = report["video_level"]["per_class"]
    x = np.arange(len(TARGET_CLASSES))
    fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for offset, metric, label in ((-0.25, "precision", "Precision"), (0, "recall", "Recall"),
                                  (0.25, "f1-score", "F1")):
        axis.bar(x + offset, [metrics[name][metric] for name in TARGET_CLASSES], width=0.24, label=label)
    axis.set_xticks(x, TARGET_CLASSES)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.legend()
    fig.savefig(output_dir / "per_class_metrics.png", dpi=140)
    plt.close(fig)

    cameras = report["by_camera"]
    fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    axis.bar(list(cameras), [values["accuracy"] for values in cameras.values()])
    axis.axhline(report["video_level"]["accuracy"], color="black", linestyle="--", linewidth=1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Camera")
    axis.set_ylabel("Accuracy")
    fig.savefig(output_dir / "camera_accuracy.png", dpi=140)
    plt.close(fig)
