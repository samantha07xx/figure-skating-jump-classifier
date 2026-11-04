"""Content-addressed experiment records; deliberately does not change RNG/backend state."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, root: Path = PROJECT_ROOT) -> dict:
    path = Path(path).resolve()
    try:
        name = path.relative_to(root.resolve()).as_posix()
    except ValueError:
        name = str(path)
    return {"path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_split_file(path: Path) -> list[dict[str, str]]:
    """Validate all partitions before a research run without changing row order."""
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or {row["split"] for row in rows} != {"train", "val", "test"}:
        raise ValueError("split must contain nonempty train, val and test partitions")
    groups = defaultdict(set)
    labels = defaultdict(set)
    from fs_jump3d.dataset import TARGET_CLASSES
    for row in rows:
        groups[row["group_id"]].add(row["split"])
        labels[row["group_id"]].add(row["jump_type"])
        if row["jump_type"] not in TARGET_CLASSES:
            raise ValueError("split contains an unsupported label")
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("physical-attempt group leaks across splits")
    if any(len(values) != 1 for values in labels.values()):
        raise ValueError("physical-attempt group has inconsistent labels")
    if len({row["filepath"] for row in rows}) != len(rows):
        raise ValueError("duplicate filepath in split")
    return rows


def environment_record() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": dict(sorted((d.metadata["Name"], d.version)
                                 for d in importlib.metadata.distributions())),
        "torch": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "mps_available": torch.backends.mps.is_available(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "numerical_environment": {key: os.environ.get(key) for key in
                                  ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
                                   "MKL_NUM_THREADS", "PYTORCH_ENABLE_MPS_FALLBACK")},
    }


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *args],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture_run(output_dir: Path, *, split_csv: Path, data_root: Path,
                config: dict, cache_dir: Path | None = None,
                checkpoints: tuple[Path, ...] = (),
                partitions: tuple[str, ...] = ("train", "val", "test"),
                record_kind: str = "snapshot") -> Path:
    """Record exact input bytes and source files before execution.

    Hashing is read-only. Existing caches are recorded, never invalidated or
    regenerated here. Missing caches are recorded as cold-cache inputs.
    """
    if record_kind not in ("snapshot", "experiment"):
        raise ValueError("record_kind must be snapshot or experiment")
    rows = validate_split_file(split_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to replace provenance: {manifest_path}")
    sources = []
    for directory in ("src", "scripts", "configs", "tests", "web", "docs", "environments", ".github"):
        sources.extend(p for p in (PROJECT_ROOT / directory).rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts)
    sources.extend(PROJECT_ROOT / name for name in
                   ("README.md", "requirements.txt", "pyproject.toml", ".gitignore", "LICENSE", "CITATION.cff")
                   if (PROJECT_ROOT / name).is_file())
    source_records = []
    for source in sorted(sources):
        destination = output_dir / "source" / source.relative_to(PROJECT_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_records.append(file_record(source))
    shutil.copyfile(split_csv, output_dir / "split.csv")
    inputs = [file_record(Path(split_csv))]
    inputs.extend(file_record(Path(checkpoint)) for checkpoint in checkpoints)
    cache_records, missing_cache = [], []
    for row in rows:
        if row["split"] not in partitions:
            continue
        relative = Path(row["filepath"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe dataset-relative path: {relative}")
        inputs.append(file_record(Path(data_root) / relative))
        if cache_dir is not None:
            cache = (Path(cache_dir) / f"{config['frames_per_clip']}x{config['image_size']}"
                     / relative.with_suffix(".npy"))
            if cache.exists():
                cache_records.append(file_record(cache))
            else:
                missing_cache.append(str(cache.resolve()))
    manifest = {
        "schema_version": 2,
        "record_kind": record_kind,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root_at_capture": str(PROJECT_ROOT),
        "command": sys.argv,
        "working_directory": str(Path.cwd()),
        "git": {"commit": _git("rev-parse", "HEAD"),
                "status": _git("status", "--short", "--untracked-files=all")},
        "environment": environment_record(),
        "config": config,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "input_partitions": list(partitions),
        "sources": source_records,
        "inputs": inputs,
        "existing_caches": cache_records,
        "missing_caches_at_start": missing_cache,
        "notes": ["Captured now; not evidence of a historical training environment.",
                  "RNG and backend settings are observed, not changed.",
                  "Do not mutate source, data or caches during execution.",
                  "Exact cross-device or cross-version numerical equality is not guaranteed."],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def verify_manifest(manifest_path: Path, root: Path = PROJECT_ROOT, *, check_environment: bool = True) -> list[str]:
    """Verify available input/source bytes; report every mismatch."""
    manifest = json.loads(Path(manifest_path).read_text())
    errors = []
    if manifest.get("record_kind") not in ("snapshot", "experiment"):
        errors.append("legacy/unknown record kind: recapture to distinguish snapshots from completed experiments")
    if check_environment:
        current = environment_record()
        for key, value in manifest["environment"].items():
            if current.get(key) != value:
                errors.append(f"environment mismatch: {key}")
    records = manifest["sources"] + manifest["inputs"] + manifest["existing_caches"]
    outputs_path = Path(manifest_path).parent / "outputs.json"
    if outputs_path.exists():
        outputs = json.loads(outputs_path.read_text())
        if outputs["manifest_sha256"] != sha256_file(manifest_path):
            errors.append("changed: provenance manifest")
        records += outputs["outputs"]
        if not outputs["outputs"]:
            errors.append("empty experiment output record")
    elif manifest.get("record_kind") == "experiment":
        errors.append("incomplete experiment: missing outputs.json")
    for record in records:
        path = Path(root) / record["path"]
        if not path.is_file():
            errors.append(f"missing: {record['path']}")
        elif path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            errors.append(f"changed: {record['path']}")
    for record in manifest["sources"]:
        snapshot = Path(manifest_path).parent / "source" / record["path"]
        if not snapshot.is_file() or sha256_file(snapshot) != record["sha256"]:
            errors.append(f"changed or missing source snapshot: {record['path']}")
    split_snapshot = Path(manifest_path).parent / "split.csv"
    if not split_snapshot.is_file() or sha256_file(split_snapshot) != manifest["inputs"][0]["sha256"]:
        errors.append("changed: split snapshot")
    return errors


def seal_outputs(output_dir: Path, paths: list[Path]) -> None:
    """Link completed result bytes to their input manifest."""
    output_dir = Path(output_dir)
    if not paths:
        raise ValueError("cannot seal an experiment without outputs")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["record_kind"] = "experiment"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    record = {"manifest_sha256": sha256_file(output_dir / "manifest.json"),
              "outputs": [file_record(path) for path in paths]}
    (output_dir / "outputs.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
