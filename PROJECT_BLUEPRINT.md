# Figure Skating Jump Recognition Project Blueprint

## Current Status

This project is at **Milestone 1: Dataset Inspection / Audit**.

The repository was newly created locally. Git may not be initialized, and no GitHub remote should be assumed until inspected.

The prior dataset plan referenced SkatingVerse. That plan is obsolete for the MVP. The project now uses **FS-Jump3D** RGB video data only.

## Project Goal

Build an end-to-end PyTorch machine learning project that takes an already-trimmed single-jump figure skating MP4 clip and predicts one of six jump types:

1. Axel
2. Flip
3. Loop
4. Lutz
5. Salchow
6. Toeloop

For display only, `Toeloop` may be rendered as `Toe Loop`.

The MVP does not:

- detect jumps inside full skating programs
- predict number of rotations
- score jump quality or GOE
- classify combination jumps
- use YOLO
- perform full-program temporal detection

## Planned Model Architecture

The planned modeling pipeline remains:

```text
Video
-> OpenCV preprocessing
-> uniform frame sampling
-> custom CNN spatial feature extraction
-> BiLSTM temporal modeling
-> dense classification head
-> six-class prediction + confidence
```

Use PyTorch.

## Verified Dataset Facts

Dataset: **FS-Jump3D**

Official repository: https://github.com/ryota-skating/FS-Jump3D

For the MVP, use only RGB video data. Do not use C3D or JSON pose data.

The four downloaded ZIP archives already exist under `data/raw/`:

- `videos_part1.zip` -> `skater_A`
- `videos_part2.zip` -> `skater_B`
- `videos_part3.zip` -> `skater_C`
- `videos_part4.zip` -> `skater_D`

The archives have passed `unzip -t` integrity checks before this milestone. Do not redownload them.

The extracted-data location for this repository is:

```text
data/raw/fs-jump3d/
```

This directory is ignored by Git, along with the ZIP archives. Raw videos must never be committed.

The verified real structure is:

```text
skater_A/
  cam_1/
  cam_2/
  ...
  cam_12/
```

and equivalently for `skater_B`, `skater_C`, and `skater_D`.

Real video filenames look like:

```text
skater_A/cam_1/Salchow_4.mp4
skater_A/cam_1/Salchow_10.mp4
skater_A/cam_1/Axel_10.mp4
skater_A/cam_1/Comb_1.mp4
```

Filename format:

```text
<JumpType>_<AttemptNumber>.mp4
```

Example metadata:

```text
path = skater_A/cam_1/Salchow_4.mp4
skater = A
camera = 1
jump_type = Salchow
attempt = 4
```

The ZIP files also contain macOS metadata such as `__MACOSX/`, `.DS_Store`, and `._Salchow_4.mp4`. These are not dataset samples and must always be ignored.

## Target Labels And Exclusions

Canonical internal target labels:

- Axel
- Flip
- Loop
- Lutz
- Salchow
- Toeloop

Combination jumps such as `Comb_1.mp4` and `Comb_4.mp4` must be excluded from the MVP training dataset. They may remain in audit artifacts, marked as excluded, so the data inventory is complete.

Do not assume every skater has exactly the same number of attempts. Determine counts by scanning the real dataset.

## Critical Multi-View Leakage Rule

FS-Jump3D is multi-view. The same physical jump attempt is recorded by multiple cameras.

For example:

```text
skater_A/cam_1/Salchow_4.mp4
skater_A/cam_2/Salchow_4.mp4
...
skater_A/cam_12/Salchow_4.mp4
```

represent different camera views of the same physical jump attempt.

Define the physical-attempt group ID as:

```text
<skater>_<jump_type>_<attempt_number>
```

Example:

```text
A_Salchow_4
```

All camera views belonging to the same group must always remain in the same split. Never use naive video-level random splitting.

## Repository Structure

```text
configs/                 Project configuration files
data/
  audit/                 Small reproducible audit artifacts tracked in Git
  raw/                   Ignored ZIP archives and extracted raw dataset
models/                  Ignored trained weights/checkpoints
notebooks/               Exploratory notebooks, if needed
reports/                 Ignored large generated reports/figures
scripts/                 Reproducible command-line scripts
src/fs_jump3d/           Project Python package
tests/                   Unit tests
uploads/                 Ignored user-uploaded inference videos
```

## Milestones

### Milestone 1: Dataset Inspection / Audit

Status: current milestone.

Required outputs:

- repository scaffold
- FS-Jump3D-aware documentation
- ignored raw-data paths
- dataset path parser
- reproducible audit script
- tests for parser and key audit logic
- `data/audit/dataset_index.csv`
- `data/audit/dataset_summary.md`

Dataset index columns should include at minimum:

- filepath
- skater
- camera
- jump_type
- attempt_number
- group_id
- is_combination
- split

The `split` column remains empty in Milestone 1.

Audit reporting should include:

- total number of real MP4 files
- number of physical jump attempts
- number of usable six-class MP4 files
- number of excluded combination-jump videos
- counts by jump class
- counts by skater
- counts by camera
- counts by skater x class
- number of camera views per physical attempt
- missing camera views, if any
- duplicate or suspicious paths
- malformed/unexpected filenames
- unreadable/corrupt MP4 files if practical to verify
- unexpected jump labels
- whether the actual dataset matches the verified structure described above

### Milestone 2: Grouped Train/Validation/Test Split

Create reproducible split logic that groups by physical-attempt ID. Do not split camera views from the same physical attempt across different splits.

Recommended outputs:

- updated dataset index with split assignments
- split summary tables
- tests proving group integrity

### Milestone 3: Video Preprocessing

Implement OpenCV-based preprocessing for already-trimmed MP4 clips:

- video loading
- fixed-length uniform frame sampling
- resizing/cropping
- normalization
- optional lightweight augmentation for training

### Milestone 4: Model Training

Implement the custom CNN + BiLSTM + dense classification head in PyTorch, plus training loops, checkpointing, and experiment configuration.

### Milestone 5: Evaluation

Evaluate on the held-out grouped test set:

- accuracy
- per-class precision/recall/F1
- confusion matrix
- qualitative failure examples

### Milestone 6: Inference

Create an inference entrypoint for one already-trimmed MP4 clip that returns:

- predicted jump class
- confidence
- class probability distribution

### Milestone 7: Web MVP

Build a minimal interface for uploading or selecting one already-trimmed clip and displaying the six-class prediction. User-uploaded videos must remain ignored by Git.

## Git Strategy

- Inspect Git state before assuming initialization or remotes.
- Keep raw videos, ZIP archives, extracted data, model weights, and large generated artifacts out of Git.
- Commit source code, tests, documentation, configuration, and small reproducible audit artifacts.
- Never force-push without explicit authorization.
- If a valid remote exists and authentication permits, push milestone commits; otherwise stop and report the local commit only.

## Open Questions For Audit Or Later Milestones

- Actual counts by class, skater, and camera.
- Whether every physical attempt has all 12 camera views.
- Whether any unexpected jump labels or malformed filenames exist.
- Whether any MP4 files are unreadable or corrupt when checked with available local video tooling.
- Best grouped split strategy after observing true class/skater balance.
