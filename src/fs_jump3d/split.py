from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math
import random
from typing import Iterable

from fs_jump3d.dataset import TARGET_CLASSES


SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_SEED = 42
EXPECTED_VIEWS_PER_GROUP = 12


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_index(index_path: Path) -> list[dict[str, str]]:
    with index_path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_rows(rows: list[dict[str, object]], output_path: Path, fieldnames: Iterable[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def ratio_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    ratio_sum = sum(ratios[split] for split in SPLIT_NAMES)
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")

    raw = {split: total * ratios[split] for split in SPLIT_NAMES}
    counts = {split: int(math.floor(raw[split])) for split in SPLIT_NAMES}
    remainder = total - sum(counts.values())
    ranked = sorted(SPLIT_NAMES, key=lambda split: (raw[split] - counts[split], -SPLIT_NAMES.index(split)), reverse=True)
    for split in ranked[:remainder]:
        counts[split] += 1
    return counts


def target_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row.copy()
        for row in rows
        if parse_bool(row.get("is_target_class")) and not parse_bool(row.get("is_combination"))
    ]


def group_target_rows(rows: list[dict[str, str]], expected_views_per_group: int = EXPECTED_VIEWS_PER_GROUP) -> dict[str, dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)

    grouped: dict[str, dict[str, object]] = {}
    for group_id, group_rows in groups.items():
        labels = {row["jump_type"] for row in group_rows}
        skaters = {row["skater"] for row in group_rows}
        cameras = {int(row["camera"]) for row in group_rows}
        if len(labels) != 1:
            raise ValueError(f"group {group_id} has multiple labels: {sorted(labels)}")
        if len(skaters) != 1:
            raise ValueError(f"group {group_id} has multiple skaters: {sorted(skaters)}")
        if labels - set(TARGET_CLASSES):
            raise ValueError(f"group {group_id} contains non-target labels: {sorted(labels)}")
        if len(group_rows) != expected_views_per_group:
            raise ValueError(
                f"group {group_id} has {len(group_rows)} views, expected {expected_views_per_group}"
            )
        if len(cameras) != expected_views_per_group:
            raise ValueError(f"group {group_id} has duplicate or missing camera views")

        grouped[group_id] = {
            "group_id": group_id,
            "jump_type": next(iter(labels)),
            "skater": next(iter(skaters)),
            "rows": sorted(group_rows, key=lambda row: int(row["camera"])),
        }

    return grouped


def assign_group_splits(
    rows: list[dict[str, str]],
    seed: int = DEFAULT_SEED,
    ratios: dict[str, float] | None = None,
    expected_views_per_group: int = EXPECTED_VIEWS_PER_GROUP,
) -> dict[str, str]:
    ratios = ratios or DEFAULT_SPLIT_RATIOS
    grouped = group_target_rows(rows, expected_views_per_group=expected_views_per_group)
    rng = random.Random(seed)

    labels = list(TARGET_CLASSES)
    skaters = sorted({str(group["skater"]) for group in grouped.values()})

    groups_by_label_skater: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group_id, group in grouped.items():
        groups_by_label_skater[(str(group["jump_type"]), str(group["skater"]))].append(group_id)

    for group_ids in groups_by_label_skater.values():
        group_ids.sort()
        rng.shuffle(group_ids)

    label_totals = Counter(str(group["jump_type"]) for group in grouped.values())
    skater_totals = Counter(str(group["skater"]) for group in grouped.values())
    label_targets = {label: ratio_counts(label_totals[label], ratios) for label in labels}
    skater_targets = {skater: ratio_counts(skater_totals[skater], ratios) for skater in skaters}

    assignments: dict[str, str] = {}
    skater_assigned = {skater: Counter() for skater in skaters}

    for split in SPLIT_NAMES:
        for label in labels:
            needed = label_targets[label][split]
            assigned_for_label = 0
            while assigned_for_label < needed:
                candidates = []
                for skater in skaters:
                    available = [
                        group_id
                        for group_id in groups_by_label_skater[(label, skater)]
                        if group_id not in assignments
                    ]
                    skater_remaining = skater_targets[skater][split] - skater_assigned[skater][split]
                    if available and skater_remaining > 0:
                        candidates.append((skater_remaining, len(available), skater))

                if not candidates:
                    raise ValueError(f"could not assign {needed} {label} groups to {split}")

                _, _, selected_skater = max(candidates, key=lambda item: (item[0], item[1], item[2]))
                selected_group = next(
                    group_id
                    for group_id in groups_by_label_skater[(label, selected_skater)]
                    if group_id not in assignments
                )
                assignments[selected_group] = split
                skater_assigned[selected_skater][split] += 1
                assigned_for_label += 1

    if len(assignments) != len(grouped):
        missing = sorted(set(grouped) - set(assignments))
        raise ValueError(f"not all target groups were assigned: {missing[:10]}")

    return assignments


def apply_splits_to_rows(
    all_rows: list[dict[str, str]],
    assignments: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_with_splits: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for row in all_rows:
        updated = row.copy()
        updated["split"] = assignments.get(row["group_id"], "")
        all_with_splits.append(updated)
        if updated["split"]:
            split_rows.append(updated)

    return all_with_splits, split_rows


def split_nested_counts(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, int]]:
    counts = {split: Counter() for split in SPLIT_NAMES}
    for row in rows:
        counts[str(row["split"])][str(row[key])] += 1
    return {split: dict(counts[split]) for split in SPLIT_NAMES}


def group_counts_by_split(rows: list[dict[str, object]]) -> dict[str, int]:
    groups_by_split = {split: set() for split in SPLIT_NAMES}
    for row in rows:
        groups_by_split[str(row["split"])].add(str(row["group_id"]))
    return {split: len(groups_by_split[split]) for split in SPLIT_NAMES}


def verify_split(split_rows: list[dict[str, object]], all_rows: list[dict[str, str]]) -> dict[str, object]:
    target = target_rows(all_rows)
    group_to_splits: dict[str, set[str]] = defaultdict(set)
    group_to_cameras: dict[str, set[int]] = defaultdict(set)
    for row in split_rows:
        group_to_splits[str(row["group_id"])].add(str(row["split"]))
        group_to_cameras[str(row["group_id"])].add(int(row["camera"]))

    filepath_counts = Counter(str(row["filepath"]) for row in split_rows)
    labels = sorted({str(row["jump_type"]) for row in split_rows})

    return {
        "zero_group_leakage": all(len(splits) == 1 for splits in group_to_splits.values()),
        "all_groups_have_expected_views": all(
            len(cameras) == EXPECTED_VIEWS_PER_GROUP for cameras in group_to_cameras.values()
        ),
        "duplicate_filepaths": sorted(path for path, count in filepath_counts.items() if count > 1),
        "all_usable_samples_assigned_once": len(split_rows) == len(target)
        and len(filepath_counts) == len(target),
        "combination_jumps_excluded": all(not parse_bool(row.get("is_combination")) for row in split_rows),
        "only_target_labels": labels == list(TARGET_CLASSES),
        "assigned_target_rows": len(split_rows),
        "expected_target_rows": len(target),
        "assigned_groups": len(group_to_splits),
        "expected_groups": len({row["group_id"] for row in target}),
    }


def summarize_split(
    split_rows: list[dict[str, object]],
    all_rows: list[dict[str, str]],
    seed: int,
    ratios: dict[str, float],
) -> dict[str, object]:
    group_counts = group_counts_by_split(split_rows)
    video_counts = dict(Counter(str(row["split"]) for row in split_rows))
    class_by_split = split_nested_counts(split_rows, "jump_type")
    skater_by_split = split_nested_counts(split_rows, "skater")

    class_split_distribution = {
        label: {
            split: sum(1 for row in split_rows if row["jump_type"] == label and row["split"] == split)
            for split in SPLIT_NAMES
        }
        for label in TARGET_CLASSES
    }
    skater_split_distribution = {
        skater: {
            split: sum(1 for row in split_rows if row["skater"] == skater and row["split"] == split)
            for split in SPLIT_NAMES
        }
        for skater in sorted({str(row["skater"]) for row in split_rows})
    }

    group_class_distribution = {
        label: {
            split: class_split_distribution[label][split] // EXPECTED_VIEWS_PER_GROUP
            for split in SPLIT_NAMES
        }
        for label in TARGET_CLASSES
    }
    group_skater_distribution = {
        skater: {
            split: skater_split_distribution[skater][split] // EXPECTED_VIEWS_PER_GROUP
            for split in SPLIT_NAMES
        }
        for skater in skater_split_distribution
    }

    return {
        "seed": seed,
        "ratios": ratios,
        "split_names": list(SPLIT_NAMES),
        "group_counts_by_split": {split: group_counts.get(split, 0) for split in SPLIT_NAMES},
        "video_counts_by_split": {split: video_counts.get(split, 0) for split in SPLIT_NAMES},
        "class_video_distribution_by_split": class_by_split,
        "skater_video_distribution_by_split": skater_by_split,
        "class_x_split_video_distribution": class_split_distribution,
        "skater_x_split_video_distribution": skater_split_distribution,
        "class_x_split_group_distribution": group_class_distribution,
        "skater_x_split_group_distribution": group_skater_distribution,
        "checks": verify_split(split_rows, all_rows),
    }


def markdown_matrix(title: str, matrix: dict[str, dict[str, int]]) -> list[str]:
    lines = [f"## {title}", "", "| Item | Train | Val | Test |", "| --- | ---: | ---: | ---: |"]
    for item in sorted(matrix):
        values = matrix[item]
        lines.append(f"| {item} | {values.get('train', 0)} | {values.get('val', 0)} | {values.get('test', 0)} |")
    lines.append("")
    return lines


def write_summary_markdown(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checks = summary["checks"]
    lines = [
        "# FS-Jump3D Grouped Split Summary",
        "",
        "## Strategy",
        "",
        "- Split level: physical-attempt `group_id`, never individual videos.",
        "- Split ratios: `70/15/15` for train/validation/test.",
        f"- Seed: `{summary['seed']}`.",
        "- Scope: six target classes only; `Comb` rows are excluded from the ML split.",
        "- Decision note: the blueprint allowed `70/15/15` or `80/10/10`; `70/15/15` was selected because 40 groups per class divides cleanly into 28/6/6 groups and gives larger validation/test sets.",
        "",
        "## Counts",
        "",
        "| Split | Physical-attempt groups | Video samples |",
        "| --- | ---: | ---: |",
    ]
    for split in SPLIT_NAMES:
        lines.append(
            f"| {split} | {summary['group_counts_by_split'][split]} | {summary['video_counts_by_split'][split]} |"
        )
    lines.extend([""])
    lines.extend(markdown_matrix("Class x Split Distribution (Groups)", summary["class_x_split_group_distribution"]))
    lines.extend(markdown_matrix("Class x Split Distribution (Videos)", summary["class_x_split_video_distribution"]))
    lines.extend(markdown_matrix("Skater x Split Distribution (Groups)", summary["skater_x_split_group_distribution"]))
    lines.extend(markdown_matrix("Skater x Split Distribution (Videos)", summary["skater_x_split_video_distribution"]))
    lines.extend(
        [
            "## Leakage And Integrity Checks",
            "",
            f"- Zero group leakage: {checks['zero_group_leakage']}",
            f"- All groups have 12 views: {checks['all_groups_have_expected_views']}",
            f"- Duplicate filepaths: {checks['duplicate_filepaths'] or 'None'}",
            f"- All usable samples assigned exactly once: {checks['all_usable_samples_assigned_once']}",
            f"- Combination jumps excluded: {checks['combination_jumps_excluded']}",
            f"- Only target labels appear: {checks['only_target_labels']}",
            f"- Assigned target rows: {checks['assigned_target_rows']}",
            f"- Expected target rows: {checks['expected_target_rows']}",
            f"- Assigned groups: {checks['assigned_groups']}",
            f"- Expected groups: {checks['expected_groups']}",
            "",
            "## Notes",
            "",
            "- This is the primary MVP group-aware split, not a skater-independent or leave-one-skater-out evaluation.",
            "- Skater distribution is balanced exactly across splits for this audited dataset, but the split still includes all four skaters in each split.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def write_summary_json(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def create_grouped_split(
    index_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    ratios: dict[str, float] | None = None,
) -> dict[str, object]:
    ratios = ratios or DEFAULT_SPLIT_RATIOS
    all_rows = read_index(index_path)
    target = target_rows(all_rows)
    assignments = assign_group_splits(target, seed=seed, ratios=ratios)
    all_with_splits, split_rows = apply_splits_to_rows(all_rows, assignments)
    summary = summarize_split(split_rows, all_rows, seed=seed, ratios=ratios)

    fieldnames = list(all_rows[0].keys()) if all_rows else []
    write_rows(all_with_splits, output_dir / "dataset_index_with_splits.csv", fieldnames)
    write_rows(split_rows, output_dir / "dataset_split.csv", fieldnames)
    write_summary_markdown(summary, output_dir / "split_summary.md")
    write_summary_json(summary, output_dir / "split_summary.json")
    return summary
