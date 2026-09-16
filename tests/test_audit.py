from pathlib import Path

from fs_jump3d.audit import build_index, summarize


def make_video(data_root: Path, relative_path: str) -> None:
    path = data_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_audit_summarizes_grouped_multiview_attempts(tmp_path: Path) -> None:
    data_root = tmp_path / "fs-jump3d"
    make_video(data_root, "skater_A/cam_1/Salchow_4.mp4")
    make_video(data_root, "skater_A/cam_2/Salchow_4.mp4")
    make_video(data_root, "skater_A/cam_1/Comb_1.mp4")
    make_video(data_root, "__MACOSX/skater_A/cam_1/._Salchow_4.mp4")

    rows, malformed, mp4_header_issues = build_index(data_root)
    summary = summarize(rows, malformed, mp4_header_issues=mp4_header_issues)

    assert summary["total_real_mp4_files"] == 3
    assert summary["usable_six_class_mp4_files"] == 2
    assert summary["excluded_combination_jump_videos"] == 1
    assert summary["physical_jump_attempts_all_labels"] == 2
    assert summary["physical_jump_attempts_six_class"] == 1
    assert summary["counts_by_jump_class"] == {"Salchow": 2}
    assert summary["views_per_physical_attempt"]["distribution"] == {1: 1, 2: 1}
    assert summary["group_id_verification"]["all_groups_camera_free"] is True
    assert len(summary["mp4_header_issues"]) == 3
