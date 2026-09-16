from pathlib import Path

import pytest

from fs_jump3d.dataset import is_metadata_path, parse_video_path


def test_parse_valid_video_path(tmp_path: Path) -> None:
    data_root = tmp_path / "fs-jump3d"
    video_path = data_root / "skater_A" / "cam_1" / "Salchow_4.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")

    parsed = parse_video_path(video_path, data_root)

    assert parsed.filepath == "skater_A/cam_1/Salchow_4.mp4"
    assert parsed.skater == "A"
    assert parsed.camera == 1
    assert parsed.jump_type == "Salchow"
    assert parsed.attempt_number == 4
    assert parsed.group_id == "A_Salchow_4"
    assert parsed.is_combination is False
    assert parsed.is_target_class is True
    assert parsed.split == ""


def test_parse_combination_video_path(tmp_path: Path) -> None:
    data_root = tmp_path / "fs-jump3d"
    video_path = data_root / "skater_C" / "cam_12" / "Comb_4.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")

    parsed = parse_video_path(video_path, data_root)

    assert parsed.jump_type == "Comb"
    assert parsed.group_id == "C_Comb_4"
    assert parsed.is_combination is True
    assert parsed.is_target_class is False


def test_metadata_paths_are_ignored() -> None:
    assert is_metadata_path(Path("__MACOSX/skater_A/cam_1/._Salchow_4.mp4"))
    assert is_metadata_path(Path("skater_A/.DS_Store"))
    assert is_metadata_path(Path("skater_A/cam_1/._Salchow_4.mp4"))


def test_parse_rejects_unexpected_layout(tmp_path: Path) -> None:
    data_root = tmp_path / "fs-jump3d"
    video_path = data_root / "skater_A" / "Salchow_4.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")

    with pytest.raises(ValueError, match="expected skater/camera/video"):
        parse_video_path(video_path, data_root)


def test_parse_rejects_unexpected_filename(tmp_path: Path) -> None:
    data_root = tmp_path / "fs-jump3d"
    video_path = data_root / "skater_A" / "cam_1" / "Salchow-final.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")

    with pytest.raises(ValueError, match="unexpected video filename"):
        parse_video_path(video_path, data_root)
