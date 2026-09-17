from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

from fs_jump3d.dataset import TARGET_CLASSES
from fs_jump3d.model import CNNBiLSTM, ModelConfig, parameter_count
from fs_jump3d.preprocessing import PreprocessConfig
from fs_jump3d.training_data import make_loaders


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Path, project_root: Path) -> dict[str, object]:
    config = yaml.safe_load(Path(path).read_text())
    for key in ("split_csv", "data_root", "run_dir", "cache_dir"):
        config[key] = str((project_root / config[key]).resolve())
    return config


def save_checkpoint(path: Path, model: CNNBiLSTM, optimizer: torch.optim.Optimizer,
                    epoch: int, val_loss: float, val_accuracy: float,
                    preprocess: PreprocessConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "label_mapping": list(TARGET_CLASSES),
        "model_config": model.metadata(),
        "preprocessing_config": asdict(preprocess),
    }, path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[CNNBiLSTM, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["label_mapping"] != list(TARGET_CLASSES):
        raise ValueError("checkpoint label mapping differs from current mapping")
    model = CNNBiLSTM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint


def run_epoch(model: CNNBiLSTM, loader, criterion, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float, float]:
    model.train(optimizer is not None)
    total_loss = 0.0
    correct = 0
    count = 0
    truths = []
    predictions = []
    for videos, labels, _ in loader:
        videos = videos.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(videos)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise FloatingPointError("non-finite gradient")
            optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.detach()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum())
        truths.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        count += batch_size
    return total_loss / count, correct / count, float(f1_score(truths, predictions, labels=range(len(TARGET_CLASSES)), average="macro", zero_division=0))


def build_optimizer(model: CNNBiLSTM, config: dict[str, object]) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "Adam"))
    lr = float(config["learning_rate"])
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(config.get("weight_decay", 0.0)))
    raise ValueError(f"unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, object]):
    name = config.get("scheduler", "none")
    if name == "none":
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(config.get("scheduler_factor", 0.5)),
            patience=int(config.get("scheduler_patience", 2)), min_lr=1e-6)
    raise ValueError(f"unsupported scheduler: {name}")


def train(config: dict[str, object], smoke: bool = False) -> dict[str, object]:
    seed_everything(int(config["seed"]))
    device = select_device(str(config["device"]))
    preprocess = PreprocessConfig(int(config["frames_per_clip"]), int(config["image_size"]))
    model = CNNBiLSTM(ModelConfig(num_classes=int(config["num_classes"]), **config["model"])).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = torch.nn.CrossEntropyLoss()
    run_dir = Path(config["run_dir"])
    if smoke:
        run_dir = run_dir / "smoke"
    if (run_dir / "summary.json").exists():
        raise FileExistsError(f"run directory already contains a training summary: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = {**config, "actual_device": str(device), "parameter_count": parameter_count(model), "smoke": smoke}
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    train_loader, val_loader = make_loaders(
        Path(config["split_csv"]), Path(config["data_root"]), preprocess,
        int(config["batch_size"]), int(config["num_workers"]), int(config["seed"]),
        max_samples=4 if smoke else None,
        cache_dir=Path(config["cache_dir"]) if config.get("cache_dir") else None,
        augment_train=bool(config.get("augment_train", False)),
    )
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_accuracy = 0.0
    stale = 0
    start = time.monotonic()
    epochs = 1 if smoke else int(config["epochs"])
    for epoch in range(1, epochs + 1):
        epoch_start = time.monotonic()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy, train_f1 = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy, val_f1 = run_epoch(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
               "train_macro_f1": train_f1, "val_loss": val_loss, "val_accuracy": val_accuracy,
               "val_macro_f1": val_f1, "learning_rate": current_lr,
               "elapsed_seconds": time.monotonic() - epoch_start}
        history.append(row)
        with (run_dir / "history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerows(history)
        if val_loss < best_loss:
            best_loss, best_epoch, best_accuracy, stale = val_loss, epoch, val_accuracy, 0
            save_checkpoint(run_dir / "best_model.pt", model, optimizer, epoch, val_loss, val_accuracy, preprocess)
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step(val_loss)
        summary = {"device": str(device), "parameter_count": parameter_count(model),
                   "epochs_trained": epoch, "best_epoch": best_epoch,
                   "best_val_loss": best_loss, "best_val_accuracy": best_accuracy,
                   "best_val_macro_f1": history[best_epoch - 1]["val_macro_f1"],
                   "early_stopping_triggered": stale >= int(config["early_stopping_patience"]),
                   "runtime_seconds": time.monotonic() - start, "final_epoch": row,
                   "smoke": smoke}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if summary["early_stopping_triggered"]:
            break
    return summary
