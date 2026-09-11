import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from .dataset import load_dataset, split_dataset
from .labels import CLASS_NAMES
from .model import build_cnn_bilstm


def train(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    x, y, _ = load_dataset(args.data_dir, args.num_frames, (args.image_size, args.image_size))
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(x, y)

    print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")
    model = build_cnn_bilstm(args.num_frames, (args.image_size, args.image_size), len(CLASS_NAMES))

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test), axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    print(report)
    (report_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    save_confusion_matrix(matrix, report_dir / "confusion_matrix.png")
    save_training_curve(history.history, report_dir / "training_curve.png")

    model.save(model_path)
    print(f"Saved model to {model_path}")


def save_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_training_curve(history: dict, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.get("accuracy", []), label="train accuracy")
    plt.plot(history.get("val_accuracy", []), label="val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNN-BiLSTM jump classifier.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model-path", default="models/jump_classifier.keras")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=160)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
