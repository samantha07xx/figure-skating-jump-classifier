from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import csv
import json
from typing import Iterable

from fs_jump3d.dataset import TARGET_CLASSES, ParsedVideoPath, is_metadata_path, parse_video_path


INDEX_COLUMNS = (
    "filepath",
    "skater",
    "camera",
    "jump_type",
    "attempt_number",
    "group_id",
    "is_combination",
    "is_target_class",
    "split",
    "file_size_bytes",
    "mp4_header_ok",
)


def iter_real_mp4_paths(data_root: Path) -> Iterable[Path]:
    for path in sorted(data_root.rglob("*.mp4")):
        try:
            relative = path.relative_to(data_root)
        except ValueError:
            continue
        if not is_metadata_path(relative):
            yield path


def count_metadata_mp4_paths(data_root: Path) -> int:
    return sum(
        1
        for path in data_root.rglob("*.mp4")
        if is_metadata_path(path.relative_to(data_root))
    )


def inspect_mp4_header(path: Path) -> str:
    """Return an issue string for clearly invalid MP4 containers, else empty."""
    size = path.stat().st_size
    if size < 12:
        return "file is too small to be a valid MP4 container"

    with path.open("rb") as file:
        header = file.read(12)
    if len(header) < 12 or header[4:8] != b"ftyp":
        return "missing MP4 ftyp box near file start"
    return ""


def build_index(data_root: Path) -> tuple[list[dict[str, object]], list[str], list[str]]:
    rows: list[dict[str, object]] = []
    malformed: list[str] = []
    mp4_header_issues: list[str] = []

    for path in iter_real_mp4_paths(data_root):
        try:
            parsed = parse_video_path(path, data_root)
        except ValueError as exc:
            malformed.append(f"{path.relative_to(data_root).as_posix()}: {exc}")
            continue

        row = asdict(parsed)
        row["file_size_bytes"] = path.stat().st_size
        header_issue = inspect_mp4_header(path)
        row["mp4_header_ok"] = not header_issue
        if header_issue:
            mp4_header_issues.append(f"{path.relative_to(data_root).as_posix()}: {header_issue}")
        rows.append(row)

    return rows, malformed, mp4_header_issues


def summarize(
    rows: list[dict[str, object]],
    malformed: list[str],
    mp4_header_issues: list[str] | None = None,
    ignored_metadata_mp4_files: int = 0,
) -> dict[str, object]:
    mp4_header_issues = mp4_header_issues or []
    target_rows = [row for row in rows if row["is_target_class"]]
    combination_rows = [row for row in rows if row["is_combination"]]
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in rows:
        groups[str(row["group_id"])].append(row)

    target_groups = {
        group_id: group_rows
        for group_id, group_rows in groups.items()
        if all(row["is_target_class"] for row in group_rows)
    }

    views_per_group = {group_id: len(group_rows) for group_id, group_rows in groups.items()}
    missing_camera_views = []
    for group_id, group_rows in sorted(groups.items()):
        cameras = sorted(int(row["camera"]) for row in group_rows)
        expected = set(range(1, 13))
        missing = sorted(expected.difference(cameras))
        if missing:
            missing_camera_views.append(
                {
                    "group_id": group_id,
                    "observed_cameras": cameras,
                    "missing_cameras": missing,
                }
            )

    path_counts = Counter(str(row["filepath"]) for row in rows)
    duplicate_paths = sorted(path for path, count in path_counts.items() if count > 1)

    unexpected_labels = sorted(
        {
            str(row["jump_type"])
            for row in rows
            if not row["is_target_class"] and not row["is_combination"]
        }
    )

    by_skater_class: dict[str, dict[str, int]] = defaultdict(dict)
    skaters = sorted({str(row["skater"]) for row in rows})
    for skater in skaters:
        for jump_type in TARGET_CLASSES:
            by_skater_class[skater][jump_type] = sum(
                1
                for row in target_rows
                if row["skater"] == skater and row["jump_type"] == jump_type
            )

    return {
        "total_real_mp4_files": len(rows),
        "ignored_metadata_mp4_files": ignored_metadata_mp4_files,
        "physical_jump_attempts_all_labels": len(groups),
        "physical_jump_attempts_six_class": len(target_groups),
        "usable_six_class_mp4_files": len(target_rows),
        "excluded_combination_jump_videos": len(combination_rows),
        "counts_by_jump_class": dict(Counter(str(row["jump_type"]) for row in target_rows)),
        "counts_by_skater": dict(Counter(str(row["skater"]) for row in rows)),
        "counts_by_camera": dict(Counter(str(row["camera"]) for row in rows)),
        "counts_by_skater_class": dict(by_skater_class),
        "views_per_physical_attempt": {
            "min": min(views_per_group.values()) if views_per_group else 0,
            "max": max(views_per_group.values()) if views_per_group else 0,
            "distribution": dict(Counter(views_per_group.values())),
        },
        "missing_camera_view_groups": missing_camera_views,
        "duplicate_paths": duplicate_paths,
        "malformed_or_unexpected_filenames": malformed,
        "mp4_header_issues": mp4_header_issues,
        "unexpected_jump_labels": unexpected_labels,
        "group_id_verification": verify_group_ids(rows),
        "structure_matches_expected": structure_matches_expected(rows, malformed),
    }


def verify_group_ids(rows: list[dict[str, object]]) -> dict[str, object]:
    mismatches = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        expected = f"{row['skater']}_{row['jump_type']}_{row['attempt_number']}"
        if row["group_id"] != expected:
            mismatches.append({"filepath": row["filepath"], "expected": expected, "actual": row["group_id"]})
        grouped[str(row["group_id"])].append(row)

    groups_with_multiple_cameras = sum(
        1 for group_rows in grouped.values() if len({row["camera"] for row in group_rows}) > 1
    )

    return {
        "mismatches": mismatches,
        "groups_with_multiple_cameras": groups_with_multiple_cameras,
        "all_groups_camera_free": not mismatches,
    }


def structure_matches_expected(rows: list[dict[str, object]], malformed: list[str]) -> bool:
    if malformed:
        return False
    skaters = {row["skater"] for row in rows}
    cameras = {int(row["camera"]) for row in rows}
    return skaters == {"A", "B", "C", "D"} and cameras == set(range(1, 13))


def write_index(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in INDEX_COLUMNS})


def markdown_table(mapping: dict[object, object], key_header: str, value_header: str) -> str:
    lines = [f"| {key_header} | {value_header} |", "| --- | ---: |"]
    for key in sorted(mapping, key=lambda item: str(item)):
        lines.append(f"| {key} | {mapping[key]} |")
    return "\n".join(lines)


def write_summary(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing_groups = summary["missing_camera_view_groups"]
    missing_preview = "None"
    if missing_groups:
        preview_rows = missing_groups[:20]
        missing_preview = "\n".join(
            f"- {row['group_id']}: missing {row['missing_cameras']} (observed {row['observed_cameras']})"
            for row in preview_rows
        )
        if len(missing_groups) > len(preview_rows):
            missing_preview += f"\n- ... {len(missing_groups) - len(preview_rows)} more groups"

    skater_class = summary["counts_by_skater_class"]
    skater_class_lines = ["| Skater | Axel | Flip | Loop | Lutz | Salchow | Toeloop |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for skater in sorted(skater_class):
        counts = skater_class[skater]
        skater_class_lines.append(
            f"| {skater} | {counts['Axel']} | {counts['Flip']} | {counts['Loop']} | {counts['Lutz']} | {counts['Salchow']} | {counts['Toeloop']} |"
        )

    lines = [
        "# FS-Jump3D Dataset Audit",
        "",
        "## Headline Counts",
        "",
        f"- Total real MP4 files: {summary['total_real_mp4_files']}",
        f"- Ignored metadata MP4 sidecars: {summary['ignored_metadata_mp4_files']}",
        f"- Physical jump attempts, all labels: {summary['physical_jump_attempts_all_labels']}",
        f"- Physical jump attempts, six-class MVP only: {summary['physical_jump_attempts_six_class']}",
        f"- Usable six-class MP4 files: {summary['usable_six_class_mp4_files']}",
        f"- Excluded combination-jump videos: {summary['excluded_combination_jump_videos']}",
        f"- Structure matches expected skater/camera layout: {summary['structure_matches_expected']}",
        "",
        "## Counts By Jump Class",
        "",
        markdown_table(summary["counts_by_jump_class"], "Jump class", "Videos"),
        "",
        "## Counts By Skater",
        "",
        markdown_table(summary["counts_by_skater"], "Skater", "Videos"),
        "",
        "## Counts By Camera",
        "",
        markdown_table(summary["counts_by_camera"], "Camera", "Videos"),
        "",
        "## Counts By Skater And Class",
        "",
        "\n".join(skater_class_lines),
        "",
        "## Physical Attempt Camera Views",
        "",
        f"- Minimum views per group: {summary['views_per_physical_attempt']['min']}",
        f"- Maximum views per group: {summary['views_per_physical_attempt']['max']}",
        f"- View-count distribution: {summary['views_per_physical_attempt']['distribution']}",
        "",
        "## Missing Camera Views",
        "",
        missing_preview,
        "",
        "## Data Quality Checks",
        "",
        f"- Duplicate paths: {summary['duplicate_paths'] or 'None'}",
        f"- Malformed/unexpected filenames: {summary['malformed_or_unexpected_filenames'] or 'None'}",
        f"- MP4 header issues: {summary['mp4_header_issues'] or 'None'}",
        f"- Unexpected jump labels: {summary['unexpected_jump_labels'] or 'None'}",
        f"- Group ID verification: {summary['group_id_verification']}",
        "",
        "## Notes",
        "",
        "- macOS metadata paths under `__MACOSX`, `.DS_Store`, and `._*` are ignored.",
        "- `Comb` videos are indexed but excluded from the six-class MVP dataset.",
        "- The `split` column is intentionally empty in Milestone 1.",
    ]

    output_path.write_text("\n".join(lines) + "\n")


def write_json(summary: dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def run_audit(data_root: Path, output_dir: Path) -> dict[str, object]:
    rows, malformed, mp4_header_issues = build_index(data_root)
    summary = summarize(
        rows,
        malformed,
        mp4_header_issues=mp4_header_issues,
        ignored_metadata_mp4_files=count_metadata_mp4_paths(data_root),
    )
    write_index(rows, output_dir / "dataset_index.csv")
    write_summary(summary, output_dir / "dataset_summary.md")
    write_json(summary, output_dir / "dataset_summary.json")
    return summary
