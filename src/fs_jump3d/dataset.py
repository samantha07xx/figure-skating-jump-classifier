from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


TARGET_CLASSES = ("Axel", "Flip", "Loop", "Lutz", "Salchow", "Toeloop")
COMBINATION_LABELS = ("Comb",)

_SKATER_RE = re.compile(r"^skater_([A-Za-z0-9]+)$")
_CAMERA_RE = re.compile(r"^cam_(\d+)$")
_VIDEO_RE = re.compile(r"^(?P<label>[A-Za-z]+)_(?P<attempt>\d+)\.mp4$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedVideoPath:
    filepath: str
    skater: str
    camera: int
    jump_type: str
    attempt_number: int
    group_id: str
    is_combination: bool
    is_target_class: bool
    split: str = ""


def is_metadata_path(path: Path) -> bool:
    """Return True for macOS metadata paths that are not dataset samples."""
    return any(part == "__MACOSX" for part in path.parts) or path.name.startswith("._") or path.name == ".DS_Store"


def canonicalize_jump_label(label: str) -> str:
    normalized = label.strip().lower()
    for candidate in TARGET_CLASSES + COMBINATION_LABELS:
        if normalized == candidate.lower():
            return candidate
    return label


def parse_video_path(path: Path, data_root: Path) -> ParsedVideoPath:
    """Parse an FS-Jump3D RGB video path.

    Expected relative layout:
        skater_A/cam_1/Salchow_4.mp4
    """
    data_root = data_root.resolve()
    path = path.resolve()

    try:
        relative = path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"{path} is not inside dataset root {data_root}") from exc

    if is_metadata_path(relative):
        raise ValueError(f"metadata path is not a dataset sample: {relative}")

    if len(relative.parts) != 3:
        raise ValueError(f"expected skater/camera/video.mp4 path, got: {relative}")

    skater_part, camera_part, filename = relative.parts
    skater_match = _SKATER_RE.fullmatch(skater_part)
    camera_match = _CAMERA_RE.fullmatch(camera_part)
    video_match = _VIDEO_RE.fullmatch(filename)

    if skater_match is None:
        raise ValueError(f"unexpected skater directory: {skater_part}")
    if camera_match is None:
        raise ValueError(f"unexpected camera directory: {camera_part}")
    if video_match is None:
        raise ValueError(f"unexpected video filename: {filename}")

    skater = skater_match.group(1)
    camera = int(camera_match.group(1))
    jump_type = canonicalize_jump_label(video_match.group("label"))
    attempt_number = int(video_match.group("attempt"))
    is_combination = jump_type in COMBINATION_LABELS
    is_target_class = jump_type in TARGET_CLASSES
    group_id = f"{skater}_{jump_type}_{attempt_number}"

    return ParsedVideoPath(
        filepath=relative.as_posix(),
        skater=skater,
        camera=camera,
        jump_type=jump_type,
        attempt_number=attempt_number,
        group_id=group_id,
        is_combination=is_combination,
        is_target_class=is_target_class,
    )
