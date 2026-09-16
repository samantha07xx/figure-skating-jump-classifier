from __future__ import annotations

from collections import Counter, defaultdict

from fs_jump3d.dataset import TARGET_CLASSES
from fs_jump3d.split import (
    DEFAULT_SEED,
    apply_splits_to_rows,
    assign_group_splits,
    ratio_counts,
    summarize_split,
    target_rows,
)


def make_balanced_rows(groups_per_class_skater: int = 10, views_per_group: int = 12) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    skaters = ("A", "B", "C", "D")
    for skater in skaters:
        for jump_type in TARGET_CLASSES:
            for attempt in range(1, groups_per_class_skater + 1):
                group_id = f"{skater}_{jump_type}_{attempt}"
                for camera in range(1, views_per_group + 1):
                    rows.append(
                        {
                            "filepath": f"skater_{skater}/cam_{camera}/{jump_type}_{attempt}.mp4",
                            "skater": skater,
                            "camera": str(camera),
                            "jump_type": jump_type,
                            "attempt_number": str(attempt),
                            "group_id": group_id,
                            "is_combination": "False",
                            "is_target_class": "True",
                            "split": "",
                            "file_size_bytes": "100",
                            "mp4_header_ok": "True",
                        }
                    )

    for camera in range(1, views_per_group + 1):
        rows.append(
            {
                "filepath": f"skater_A/cam_{camera}/Comb_1.mp4",
                "skater": "A",
                "camera": str(camera),
                "jump_type": "Comb",
                "attempt_number": "1",
                "group_id": "A_Comb_1",
                "is_combination": "True",
                "is_target_class": "False",
                "split": "",
                "file_size_bytes": "100",
                "mp4_header_ok": "True",
            }
        )
    return rows


def test_ratio_counts_for_milestone_two_default() -> None:
    assert ratio_counts(40, {"train": 0.70, "val": 0.15, "test": 0.15}) == {
        "train": 28,
        "val": 6,
        "test": 6,
    }


def test_grouped_split_is_deterministic_and_leak_free() -> None:
    rows = make_balanced_rows()
    target = target_rows(rows)

    first = assign_group_splits(target, seed=DEFAULT_SEED)
    second = assign_group_splits(target, seed=DEFAULT_SEED)

    assert first == second

    _, split_rows = apply_splits_to_rows(rows, first)
    summary = summarize_split(split_rows, rows, seed=DEFAULT_SEED, ratios={"train": 0.70, "val": 0.15, "test": 0.15})

    assert summary["group_counts_by_split"] == {"train": 168, "val": 36, "test": 36}
    assert summary["video_counts_by_split"] == {"train": 2016, "val": 432, "test": 432}
    assert summary["checks"]["zero_group_leakage"] is True
    assert summary["checks"]["all_groups_have_expected_views"] is True
    assert summary["checks"]["all_usable_samples_assigned_once"] is True
    assert summary["checks"]["combination_jumps_excluded"] is True
    assert summary["checks"]["only_target_labels"] is True


def test_all_views_for_each_group_share_one_split() -> None:
    rows = make_balanced_rows()
    assignments = assign_group_splits(target_rows(rows), seed=DEFAULT_SEED)
    _, split_rows = apply_splits_to_rows(rows, assignments)

    splits_by_group: dict[str, set[str]] = defaultdict(set)
    cameras_by_group: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        splits_by_group[row["group_id"]].add(row["split"])
        cameras_by_group[row["group_id"]].add(row["camera"])

    assert all(len(splits) == 1 for splits in splits_by_group.values())
    assert all(len(cameras) == 12 for cameras in cameras_by_group.values())


def test_class_balance_is_exact_for_audited_dataset_shape() -> None:
    rows = make_balanced_rows()
    assignments = assign_group_splits(target_rows(rows), seed=DEFAULT_SEED)

    split_by_group = assignments
    class_split_counts = {label: Counter() for label in TARGET_CLASSES}
    for group_id, split in split_by_group.items():
        jump_type = group_id.split("_")[1]
        class_split_counts[jump_type][split] += 1

    for jump_type in TARGET_CLASSES:
        assert class_split_counts[jump_type] == {"train": 28, "val": 6, "test": 6}


def test_skater_distribution_is_balanced_for_primary_mvp_split() -> None:
    rows = make_balanced_rows()
    assignments = assign_group_splits(target_rows(rows), seed=DEFAULT_SEED)

    skater_split_counts = {skater: Counter() for skater in ("A", "B", "C", "D")}
    for group_id, split in assignments.items():
        skater = group_id.split("_")[0]
        skater_split_counts[skater][split] += 1

    for skater in ("A", "B", "C", "D"):
        assert skater_split_counts[skater] == {"train": 42, "val": 9, "test": 9}
