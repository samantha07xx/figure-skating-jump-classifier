# Figure Skating Jump Recognition

End-to-end PyTorch project for classifying already-trimmed single-jump figure skating MP4 clips into six jump types: Axel, Flip, Loop, Lutz, Salchow, and Toeloop.

This repository is currently stopped after **Milestone 2: Grouped Train/Validation/Test Split**. It does not yet implement preprocessing, model training, evaluation, inference, or a web application.

## Dataset

The MVP uses the RGB video portion of FS-Jump3D:

https://github.com/ryota-skating/FS-Jump3D

The four downloaded archives are expected locally under `data/raw/`:

- `videos_part1.zip` -> `skater_A`
- `videos_part2.zip` -> `skater_B`
- `videos_part3.zip` -> `skater_C`
- `videos_part4.zip` -> `skater_D`

The extracted dataset root used by Milestone 1 is:

`data/raw/fs-jump3d/`

Raw videos, ZIP archives, and extracted dataset folders are ignored by Git and must not be committed.

## Milestone 1 Audit

Run the audit from the repository root:

```bash
python3 scripts/audit_dataset.py --data-root data/raw/fs-jump3d --output-dir data/audit
```

This writes:

- `data/audit/dataset_index.csv`
- `data/audit/dataset_summary.md`

The index includes every real MP4 sample discovered under the extracted RGB videos, including combination jumps marked as excluded from the six-class MVP.

## Leakage Rule

FS-Jump3D is multi-view. Multiple cameras record the same physical jump attempt, so splitting must never happen at the individual-video level.

The physical-attempt group ID is:

```text
<skater>_<jump_type>_<attempt_number>
```

Example:

```text
A_Salchow_4
```

All camera views for the same group must remain in the same split.

## Milestone 2 Split

Run the grouped split from the repository root:

```bash
python3 scripts/create_split.py --index-path data/audit/dataset_index.csv --output-dir data/splits
```

Milestone 2 uses a deterministic `70/15/15` train/validation/test split with seed `42`, assigned at the physical-attempt `group_id` level. Combination jumps are excluded from the ML split.

This writes:

- `data/splits/dataset_split.csv`
- `data/splits/dataset_index_with_splits.csv`
- `data/splits/split_summary.md`
- `data/splits/split_summary.json`

## Tests

Run tests with:

```bash
python3 -m pytest
```
