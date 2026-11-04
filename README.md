# Figure Skating Jump Recognition

CNN-BiLSTM classification of six figure-skating jumps from trimmed RGB video clips: Axel, Flip, Loop, Lutz, Salchow, and Toe Loop. This repository contains data preparation, training, evaluation, and single-video inference code, with a FastAPI web demo.

## Method

OpenCV uniformly samples 32 frames per clip, resizes them to 224 × 224, converts them to RGB, and normalizes pixel values to `[0, 1]`. A shared CNN encodes each frame; a bidirectional LSTM models the sequence; a classifier outputs scores for the six jump classes.

Experiments compare the baseline with clip-consistent spatial and photometric augmentation, AdamW, and learning-rate scheduling. Augmentation is applied only during training. The task assumes an already-trimmed single jump; jump detection in full programs and jump-quality scoring are outside its scope.

## Dataset and experimental protocol

The six-class subset of [FS-Jump3D](https://github.com/ryota-skating/FS-Jump3D) used here contains 2,880 videos from 240 physical jump attempts by four skaters, with 12 camera views per attempt. Combination jumps are excluded.

Data are split by physical attempt, keeping all camera views of an attempt in one partition:

| Partition | Physical attempts | Videos |
| --- | ---: | ---: |
| Training | 168 | 2,016 |
| Validation | 36 | 432 |
| Test | 36 | 432 |

Experiments use seed 42, the same split and model architecture, a 30-epoch limit, and early stopping with patience 5. Each run's checkpoint is selected by minimum validation loss. Candidate methods are compared on validation data before the selected augmentation model is evaluated on the held-out test set.

All four skaters appear in each partition. This protocol evaluates held-out attempts under the dataset's recording conditions; it does not measure unseen-skater or uncontrolled-video generalization. Results are from single-seed runs.

## Results

Accuracy and macro F1 below are computed per video, treating each camera view as one prediction.

| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: | ---: |
| CNN-BiLSTM baseline | 67.13% | 0.673 | 65.97% | 0.652 |
| CNN-BiLSTM + training-only augmentation | 70.37% | 0.701 | 71.99% | 0.719 |

The [experiment comparison](data/experiments/improvement_phase1/comparison.md) reports all candidate runs and validation-based selection. The [test report](data/evaluation/phase1_augmentation_final/final_comparison.md) includes per-class metrics, camera and skater breakdowns, error analysis, and a separate multi-view evaluation that averages probabilities across an attempt's camera views. Prediction files and confusion matrices are available under `data/evaluation/`.

## Setup and reproduction

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

Raw videos and model checkpoints are not included in Git. The [reproducibility guide](docs/REPRODUCIBILITY.md) documents data preparation, the expected directory layout, training, validation, test evaluation, and artifact verification. A pinned snapshot of the current environment is available in [`environments/requirements-current.txt`](environments/requirements-current.txt).

Train into a new directory for each run:

```bash
.venv/bin/python scripts/train.py --config configs/train.yaml --run-dir models/runs/reproduction/baseline
.venv/bin/python scripts/train.py --config configs/improvement_phase1_augmentation.yaml --run-dir models/runs/reproduction/augmentation
```

Each run saves its configuration, training history, selected checkpoint, source snapshot, environment metadata, and input/result fingerprints.

## Inference demo

The web demo uses the evaluated augmentation checkpoint specified in `configs/inference.yaml`; the baseline checkpoint is preserved separately. Place the matching checkpoint at the configured path before starting the server.

```bash
.venv/bin/uvicorn --app-dir src fs_jump3d.web:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and upload one trimmed single-jump MP4. The interface displays the predicted class and softmax probabilities. The backend loads the model at startup and deletes temporary uploads after each request. The default upload limit is 100 MB.

For command-line prediction using the same checkpoint:

```bash
.venv/bin/python scripts/predict_video.py path/to/trimmed_jump.mp4
```

## Repository structure

| Path | Contents |
| --- | --- |
| `src/fs_jump3d/` | Data processing, model, training, evaluation, and inference modules |
| `scripts/` | Command-line entry points |
| `configs/` | Experiment settings and protocol descriptions |
| `data/` | Tracked audit, split, and experiment results; local raw data and caches |
| `tests/` | Automated checks |
| `web/` | Web demo interface |
| `docs/` | Reproduction instructions |
| `environments/` | Dependency snapshot |
