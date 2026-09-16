# Figure Skating Jump Recognition Project Blueprint

## 1. Project Goal

Build an end-to-end machine learning project that classifies an already-trimmed single-jump figure skating MP4 clip into one of six jump types:

1. Axel
2. Flip
3. Loop
4. Lutz
5. Salchow
6. Toeloop

The project should be reproducible, testable, and organized as a real engineering portfolio project. The intended production-facing path is:

```text
trimmed MP4 clip
-> OpenCV video loading
-> uniform frame sampling
-> custom CNN frame encoder
-> BiLSTM temporal model
-> dense classification head
-> predicted jump class + confidence
```

The model framework is PyTorch. The MVP is intentionally scoped around already-trimmed single jumps so the first complete version focuses on clean data handling, leakage-safe evaluation, and a working six-class classifier.

## 2. Current MVP Scope

The MVP accepts one already-trimmed video clip containing one primary figure skating jump and predicts one of the six canonical target classes. The input is expected to be an MP4 clip, not a full competition program.

The MVP includes:

- FS-Jump3D RGB video dataset indexing and audit
- exclusion of non-target combination jump clips
- group-aware train/validation/test splitting by physical jump attempt
- OpenCV-based frame loading and preprocessing
- a custom CNN-BiLSTM classifier
- training, validation, checkpointing, and evaluation scripts
- saved model artifacts and class mapping metadata
- single-video inference
- a minimal web interface for upload/prediction

The current repository state has completed Milestone 1 only: dataset inspection, audit artifacts, repository scaffolding, parser tests, and documentation.

## 3. Out-of-Scope Features

The MVP does not:

- detect jumps inside full skating programs
- classify full programs with multiple jumps
- localize the start/end frame of a jump
- predict number of rotations
- score jump quality, grade of execution, under-rotation, edge calls, or falls
- classify combination jumps
- use C3D features or JSON pose data from FS-Jump3D
- use YOLO or object detection
- perform full-program temporal detection
- guarantee generalization to arbitrary broadcast, competition, phone, or internet skating videos

These may become future extensions after the base six-class classifier is working and honestly evaluated.

## 4. Verified FS-Jump3D Dataset Structure

Dataset: **FS-Jump3D**

Official repository: https://github.com/ryota-skating/FS-Jump3D

The MVP uses only the RGB video data. It does not use the C3D or JSON pose data.

The four downloaded ZIP archives exist locally under `data/raw/`:

- `videos_part1.zip` -> `skater_A`
- `videos_part2.zip` -> `skater_B`
- `videos_part3.zip` -> `skater_C`
- `videos_part4.zip` -> `skater_D`

The archives were already verified with `unzip -t`. Do not redownload the dataset unless explicitly requested.

The extracted dataset root for this repository is:

```text
data/raw/fs-jump3d/
```

This directory is ignored by Git. The raw ZIP archives, extracted videos, macOS metadata folders, and video files must remain untracked.

Verified extracted layout:

```text
data/raw/fs-jump3d/
  skater_A/
    cam_1/
    cam_2/
    ...
    cam_12/
  skater_B/
    cam_1/
    ...
    cam_12/
  skater_C/
    cam_1/
    ...
    cam_12/
  skater_D/
    cam_1/
    ...
    cam_12/
```

Real video filenames follow:

```text
<JumpType>_<AttemptNumber>.mp4
```

Example:

```text
skater_A/cam_1/Salchow_4.mp4
```

Parsed metadata:

```text
skater = A
camera = 1
jump_type = Salchow
attempt_number = 4
group_id = A_Salchow_4
```

The ZIP archives also include macOS metadata such as:

- `__MACOSX/`
- `.DS_Store`
- `._Salchow_4.mp4`

These are not dataset samples and must always be ignored.

## 5. Dataset Audit Findings

Milestone 1 generated reproducible audit artifacts under `data/audit/`:

- `data/audit/dataset_index.csv`
- `data/audit/dataset_summary.md`
- `data/audit/dataset_summary.json`

Verified Milestone 1 results:

- total real MP4 files: `3,036`
- total physical attempts across all labels: `253`
- usable six-class MP4 files: `2,880`
- usable six-class physical attempts: `240`
- excluded combination-jump videos: `156`
- combination physical attempts: `13`
- target-class video count per class: `480`
- every physical attempt has exactly `12` camera views
- missing camera views: `0`
- malformed filenames: `0`
- unexpected labels: `0`
- duplicate paths: `0`
- basic MP4 header issues: `0`
- ignored metadata MP4 sidecars: `3,036`

Six-class distribution:

| Class | Videos | Physical attempts |
| --- | ---: | ---: |
| Axel | 480 | 40 |
| Flip | 480 | 40 |
| Loop | 480 | 40 |
| Lutz | 480 | 40 |
| Salchow | 480 | 40 |
| Toeloop | 480 | 40 |

Counts by skater, including combination videos:

| Skater | Videos |
| --- | ---: |
| A | 756 |
| B | 756 |
| C | 768 |
| D | 756 |

Each camera has `253` real MP4 samples, one view for each physical attempt.

## 6. Label Mapping

Canonical internal target labels:

```text
0 -> Axel
1 -> Flip
2 -> Loop
3 -> Lutz
4 -> Salchow
5 -> Toeloop
```

`Toeloop` is the canonical internal spelling because that is the spelling observed in the dataset filenames. For display only, the UI may render it as `Toe Loop`.

`Comb` is a valid observed dataset label but is not an MVP target class. Combination jump clips must be excluded from training, validation, testing, and six-class metrics. They may remain in audit outputs with `is_combination = True` for traceability.

All scripts should use a single source of truth for class names. The class order must be saved with trained model artifacts so inference uses the same mapping as training.

## 7. Data Filtering And Cleaning

Dataset loading should apply these filtering rules:

- include only real `.mp4` files under `skater_*/cam_*/`
- ignore paths under `__MACOSX/`
- ignore `.DS_Store`
- ignore AppleDouble metadata files beginning with `._`
- parse only filenames matching `<JumpType>_<AttemptNumber>.mp4`
- keep `Comb` in audit artifacts but exclude it from the MVP six-class dataset
- reject or report unexpected labels
- reject or report malformed filenames
- avoid copying or duplicating raw videos
- never commit raw videos, ZIP archives, extracted dataset directories, model weights, or uploaded inference videos

Milestone 1 performed a basic MP4 container header check. Later preprocessing should perform practical decode validation through OpenCV when frames are first read.

## 8. Critical Multi-View Leakage Rule

FS-Jump3D is multi-view. The same physical jump attempt is recorded by multiple synchronized or near-synchronized cameras.

Example:

```text
skater_A/cam_1/Salchow_4.mp4
skater_A/cam_2/Salchow_4.mp4
...
skater_A/cam_12/Salchow_4.mp4
```

These are correlated observations of the same physical action. They must not be independently randomized across train, validation, and test sets.

The physical-attempt group ID is:

```text
group_id = skater + "_" + jump_type + "_" + attempt_number
```

Example:

```text
A_Salchow_4
```

All camera views with the same `group_id` must remain in the same split. Naive video-level random splitting is prohibited because it would leak nearly identical examples across splits and overstate performance.

## 9. Dataset Index Schema

The dataset index is stored at:

```text
data/audit/dataset_index.csv
```

Current columns:

- `filepath`: project-relative path from the extracted dataset root
- `skater`: skater identifier, such as `A`
- `camera`: integer camera ID, `1` through `12`
- `jump_type`: canonical parsed label
- `attempt_number`: integer attempt number from the filename
- `group_id`: physical-attempt ID, excluding camera
- `is_combination`: boolean marker for `Comb`
- `is_target_class`: boolean marker for the six MVP labels
- `split`: empty in Milestone 1, filled in Milestone 2
- `file_size_bytes`: raw file size
- `mp4_header_ok`: boolean result from basic MP4 header validation

Milestone 2 should either update this index with split assignments or write a derived split index while preserving the original audit artifact.

## 10. Group-Aware Split Strategy

There are two separate evaluation concepts.

### A. Primary MVP Split

The primary MVP split is group-aware train/validation/test splitting at the physical-attempt `group_id` level.

Requirements:

- split only six-class target groups
- keep all 12 camera views for a `group_id` together
- preserve class balance as much as practical
- use a fixed random seed, initially `42`
- write split assignments reproducibly
- test that no `group_id` appears in more than one split
- report split counts by class, skater, camera, and group

Because each target class has `40` physical attempts and each attempt has `12` camera views, the split should operate on the `240` target physical attempts, not the `2,880` individual videos.

A reasonable initial split is `70/15/15` or `80/10/10`, chosen in Milestone 2 after reviewing group-level balance. The exact split should be documented in the generated split summary.

### B. Stronger Future Research Evaluation

A stronger future evaluation is skater-independent evaluation, including leave-one-skater-out evaluation. This measures generalization to unseen skaters rather than unseen attempts from the same four skaters.

This should be treated as a separate research evaluation because it is harder and may produce lower, more realistic performance. It is not required for the first MVP, but the project should document the distinction clearly when reporting results.

## 11. Repository Structure

Intended project structure:

```text
configs/
  dataset.yaml              Dataset paths, labels, and audit settings
  train.yaml                Future training configuration
data/
  audit/                    Small tracked audit artifacts
  raw/                      Ignored ZIP archives and extracted raw dataset
  processed/                Ignored precomputed frame tensors or derived data
models/                     Ignored trained weights and checkpoints
notebooks/                  Optional exploration
reports/                    Ignored generated figures and large reports
scripts/                    Reproducible CLI entrypoints
src/fs_jump3d/              Python package
tests/                      Unit and integration tests
uploads/                    Ignored user-uploaded inference videos
```

Tracked files should remain small and reproducible. Large generated files belong in ignored directories.

## 12. Video Preprocessing Specification

Use OpenCV for video loading and frame extraction.

Initial preprocessing defaults:

- frames per clip: `32`
- image size: `224x224`
- color format: RGB
- numeric range: `[0, 1]`
- tensor layout for model input: document and keep consistent, recommended `(T, C, H, W)` per sample before batching

Frame sampling:

- uniformly sample `32` frame indices across the full clip duration
- include the first and last usable frame when practical
- avoid assuming fixed FPS or fixed duration
- handle variable-length clips deterministically

Short-video handling:

- if a clip has fewer than `32` readable frames, repeat frames or pad using the last valid frame
- record or warn on unusually short clips
- fail clearly if no readable frame exists

Image processing:

- read frames with OpenCV, which returns BGR
- convert BGR to RGB
- resize to `224x224` initially
- normalize pixel values to `[0, 1]` as `float32`
- later model-specific normalization can be added if justified

Augmentation guidance:

- apply augmentation only to training data
- keep validation and test preprocessing deterministic
- start with conservative spatial augmentations, such as small crop/resize jitter, mild brightness/contrast changes, or horizontal flip only if label semantics remain valid
- avoid aggressive transforms that destroy skating posture, blade/body cues, or camera geometry

## 13. CNN-BiLSTM Model Architecture

The planned MVP model is a custom CNN-BiLSTM classifier.

High-level flow:

```text
batch video tensor
-> split into frame tensors
-> custom CNN frame encoder
-> frame feature vector per frame
-> sequence of frame features
-> BiLSTM temporal encoder
-> dense/dropout classifier
-> six logits
```

Components:

- custom CNN frame encoder extracts spatial features from each frame
- the same CNN weights are applied to every sampled frame
- frame features form a temporal sequence of length `32`
- BiLSTM models motion/temporal context in both directions
- classifier uses dense layers and dropout
- final output dimension is `6`
- training loss is `torch.nn.CrossEntropyLoss`

The model should output raw logits during training. Softmax probabilities are applied only for metrics, reporting, and inference display.

## 14. Training Pipeline

The training pipeline should include:

- load split-aware dataset index
- filter to `is_target_class = True`
- verify all samples have split assignments
- instantiate train, validation, and test datasets
- create PyTorch DataLoaders
- apply training-only augmentations
- build model from config
- train with CrossEntropyLoss
- evaluate on validation set each epoch
- checkpoint the best model by validation metric
- support early stopping
- save class mapping and run configuration with checkpoints
- write metrics and logs to a run directory

The training loop should be restartable and deterministic where practical. Failures should be explicit: missing files, unreadable videos, empty split, missing class labels, or group leakage should raise clear errors.

## 15. Initial Training Configuration

Initial configuration:

```yaml
seed: 42
num_classes: 6
frames_per_clip: 32
image_size: 224
batch_size: 4
epochs: 30
learning_rate: 1.0e-4
optimizer: Adam
early_stopping_patience: 5
num_workers: 2
device: auto
```

`device: auto` should select CUDA, MPS, or CPU depending on availability. The code should remain usable on CPU for smoke tests, even if full training is slow.

These values are starting points, not final claims. They should be recorded in config files and copied into each experiment artifact directory.

## 16. Evaluation Plan

Evaluate only on the held-out test split after model selection is complete.

Primary metrics:

- Accuracy
- Precision
- Recall
- F1
- macro-averaged precision, recall, and F1
- weighted precision, recall, and F1
- confusion matrix

Report per-class metrics for:

- Axel
- Flip
- Loop
- Lutz
- Salchow
- Toeloop

Evaluation should also include:

- class distribution for the evaluated split
- group count and video count
- confirmation that no `group_id` leakage occurred
- optional top-k probabilities for qualitative examples
- saved confusion matrix plot or CSV

Results must be described honestly as performance on a controlled multi-camera dataset, not as proof of broad real-world deployment readiness.

## 17. Experiment Logging And Saved Artifacts

Each training run should save a run directory containing:

- resolved training config
- Git commit hash if available
- class mapping
- split summary
- epoch-level training and validation metrics
- best checkpoint
- final checkpoint if useful
- test metrics
- confusion matrix
- notes about device and package versions

Suggested ignored artifact layout:

```text
models/runs/<timestamp_or_run_name>/
  config.yaml
  class_mapping.json
  split_summary.json
  metrics.csv
  best_model.pt
  confusion_matrix.csv
```

Only small, intentional summary artifacts should be considered for Git. Model weights and large reports remain ignored.

## 18. Inference Design

Inference input:

- one already-trimmed MP4 clip
- expected to contain one primary jump

Inference steps:

1. load saved checkpoint and class mapping
2. preprocess the video with the same deterministic preprocessing used for validation/test
3. run model in evaluation mode
4. apply softmax to logits
5. return predicted class, confidence, and full probability distribution

Inference output example:

```json
{
  "predicted_class": "Salchow",
  "confidence": 0.87,
  "probabilities": {
    "Axel": 0.01,
    "Flip": 0.03,
    "Loop": 0.04,
    "Lutz": 0.02,
    "Salchow": 0.87,
    "Toeloop": 0.03
  }
}
```

The inference path should validate file existence, extension, readability, and checkpoint compatibility.

## 19. Web MVP

The web MVP should provide a minimal local interface for:

- uploading or selecting one already-trimmed MP4
- running inference with the trained model
- displaying predicted jump type
- displaying confidence and class probabilities
- showing clear errors for invalid inputs

Uploaded videos must be stored only in ignored local directories such as `uploads/`. They must not be committed.

The UI should avoid implying that the system can analyze full programs or score jump quality. It should state through normal product wording that the input is a trimmed single-jump clip.

## 20. API Design

The first API can be simple and local. A likely design:

```text
POST /predict
```

Request:

- multipart form upload with one MP4 file

Response:

```json
{
  "predicted_class": "Axel",
  "display_class": "Axel",
  "confidence": 0.92,
  "probabilities": {
    "Axel": 0.92,
    "Flip": 0.02,
    "Loop": 0.01,
    "Lutz": 0.02,
    "Salchow": 0.01,
    "Toeloop": 0.02
  }
}
```

Possible additional endpoints:

- `GET /health`
- `GET /classes`
- `GET /model-info`

The API should keep model loading separate from request handling so tests can exercise preprocessing and prediction without running a full web server.

## 21. Dependencies

Milestone 1 dependencies:

- Python 3
- pytest

Planned project dependencies:

- PyTorch
- torchvision
- OpenCV
- NumPy
- scikit-learn
- tqdm
- PyYAML
- a lightweight web framework for the MVP, to be chosen later

The current dependency list lives in `requirements.txt`. Dependencies should be kept minimal and justified. Heavy or optional dependencies should not be introduced until the milestone that needs them.

## 22. Configuration Management

Configuration should be file-backed and reproducible.

Current config:

- `configs/dataset.yaml`

Planned configs:

- `configs/split.yaml`
- `configs/preprocess.yaml`
- `configs/train.yaml`
- `configs/inference.yaml`

Config files should define paths relative to the project root where possible. Avoid hardcoding machine-specific paths such as `/Users/samantha/...`.

Every training run should save the resolved config used for that run. Command-line arguments may override config values, but the final resolved values must be recorded.

## 23. Git/GitHub Strategy

Rules:

- inspect Git state before making repository assumptions
- keep raw data ignored
- keep extracted data ignored
- keep ZIP archives ignored
- keep model weights and checkpoints ignored
- keep uploaded inference videos ignored
- commit source code, tests, configs, docs, and small reproducible audit artifacts
- do not force-push without explicit authorization
- do not rewrite remote history without explicit authorization
- push only when a remote is configured and authentication/permissions are clear

Current state after Milestone 1:

- local Git repository exists
- no remote is configured
- raw data lives under `data/raw/fs-jump3d/`
- raw data is ignored by Git

## 24. Testing Plan

Testing should grow with each milestone.

Milestone 1 tests:

- dataset path parser
- metadata path exclusion
- combination label handling
- group ID construction
- core audit summary logic

Milestone 2 tests:

- no `group_id` appears in multiple splits
- only target-class rows are split
- split summaries match expected counts
- split output is reproducible with seed `42`

Preprocessing tests:

- short videos are padded/repeated correctly
- BGR-to-RGB conversion is applied
- output tensor shape is correct
- normalization range is `[0, 1]`
- deterministic validation/test preprocessing

Model tests:

- forward pass shape is `(batch_size, 6)`
- loss computation works
- one tiny training step runs
- checkpoint save/load preserves predictions for a fixed input

Inference/API tests:

- invalid files produce clear errors
- class mapping is loaded from artifact
- prediction response schema is stable
- web/API endpoint handles one uploaded MP4

Full training does not need to run in unit tests. Use tiny synthetic tensors and small fixtures for fast checks.

## 25. Risks And Fallbacks

Risks:

- dataset is small at the physical-attempt level despite having many camera-view videos
- multi-view correlation can inflate metrics if splitting is done incorrectly
- only four skaters are present
- model may overfit quickly
- CPU-only training may be slow
- full MP4 decoding may reveal issues not caught by header checks
- the custom CNN may underperform stronger pretrained video models
- web upload handling can accidentally create large untracked files if ignored paths are not maintained

Fallbacks:

- reduce model size for faster iteration
- start with a baseline CNN frame pooling model before BiLSTM if needed
- use stronger regularization and early stopping
- add conservative augmentation
- run skater-independent evaluation as a more realistic follow-up
- cache preprocessed tensors only in ignored directories if video decoding becomes a bottleneck

## 26. Dataset / Model Limitations

FS-Jump3D is a controlled multi-camera skating dataset with only four skaters. Its camera views are correlated observations of the same physical attempts. Strong performance on this dataset does not automatically imply strong generalization to arbitrary broadcast, competition, phone, or internet skating videos.

Known limitations:

- only four skaters
- controlled camera setup
- repeated camera views of the same attempts
- already-trimmed clips rather than full programs
- MVP ignores rotation count and jump quality
- MVP excludes combination jumps
- model will learn from RGB appearance and motion only
- no pose or biomechanical features are used in the MVP

Results should be framed as controlled-dataset performance unless future evaluations prove broader generalization.

## 27. Future Extensions

Possible future work:

- skater-independent and leave-one-skater-out evaluation
- pretrained CNN or video backbone comparisons
- pose-based features using FS-Jump3D JSON pose data
- multi-modal RGB plus pose models
- rotation-count classification
- combination jump classification
- jump detection in full programs
- temporal localization of takeoff and landing
- quality or error classification, if a suitable labeled dataset exists
- model interpretability tools such as saliency or frame contribution analysis
- improved web UI with example clips and batch processing
- export to ONNX or another deployment format

Future extensions should be added only after the core grouped-split MVP is honest and reproducible.

## 28. Complete Build Order / Milestones

### Milestone 1: Dataset Inspection / Audit

Status: complete.

Completed outputs:

- repository scaffold
- FS-Jump3D-aware blueprint
- raw-data ignore rules
- dataset path parser
- reproducible audit script
- parser and audit tests
- `data/audit/dataset_index.csv`
- `data/audit/dataset_summary.md`
- `data/audit/dataset_summary.json`

### Milestone 2: Grouped Train/Validation/Test Split

Implement group-aware split generation for the six-class target dataset.

Outputs:

- split-aware dataset index
- split summary
- tests proving no group leakage
- documentation of selected split ratios

Do not begin this milestone until explicitly approved.

### Milestone 3: Video Preprocessing

Implement OpenCV video loading and deterministic frame sampling.

Outputs:

- preprocessing module
- dataset class or loader utilities
- tests for shape, normalization, RGB conversion, and short-video behavior

### Milestone 4: CNN-BiLSTM Model

Implement the PyTorch model.

Outputs:

- custom CNN frame encoder
- BiLSTM temporal encoder
- classifier head
- forward-pass tests

### Milestone 5: Training Pipeline

Implement training and validation.

Outputs:

- training script
- config-driven training
- checkpointing
- early stopping
- metrics logging
- smoke test

### Milestone 6: Evaluation

Evaluate the selected model on the held-out grouped test set.

Outputs:

- test metrics
- per-class metrics
- macro and weighted metrics
- confusion matrix
- written evaluation report

### Milestone 7: Inference

Implement single-clip prediction.

Outputs:

- inference script
- saved class mapping support
- probability distribution output
- tests for response shape and error handling

### Milestone 8: Web MVP

Build a minimal local web application for upload and prediction.

Outputs:

- local web app
- prediction endpoint
- upload handling in ignored directory
- simple result view
- web/API smoke tests

### Milestone 9: Final Documentation And Packaging

Prepare project for review or portfolio use.

Outputs:

- updated README
- reproducible run instructions
- final limitations statement
- example inference workflow
- GitHub push if remote is configured and approved

## 29. Definition Of Done

The full MVP is done when:

- raw FS-Jump3D data is ignored and never committed
- dataset indexing is reproducible
- six-class target filtering is correct
- split generation is group-aware and tested
- no physical-attempt group leaks across splits
- preprocessing produces consistent `32 x 224 x 224` RGB inputs
- CNN-BiLSTM model trains without shape or loss errors
- best checkpoint and class mapping are saved
- held-out grouped test metrics are reported with accuracy, precision, recall, F1, macro/weighted metrics, and confusion matrix
- inference works for one trimmed MP4
- web MVP can upload/select a clip and display prediction plus confidence
- tests cover critical parser, split, preprocessing, model, and inference behavior
- README and blueprint accurately describe scope, limitations, and how to run the project
- no ZIP, MP4, extracted dataset, uploaded videos, model weights, or large binary artifacts are tracked by Git
