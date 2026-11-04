# Figure Skating Jump Recognition

A video classification project for six figure-skating jumps: **Axel, Flip, Loop, Lutz, Salchow, and Toe Loop**. It combines a custom PyTorch CNN-BiLSTM with a research pipeline for data auditing, grouped splits, controlled experiments, and detailed evaluation, plus a FastAPI web demo for single-video prediction.

The model trained with clip-consistent augmentation achieved **71.99% video-level accuracy** and **0.719 macro F1** on the held-out FS-Jump3D test split: **311 correct predictions out of 432 videos**, improving accuracy by **6.02 percentage points** over the baseline.

## Project highlights

- **Spatial and temporal modeling:** a shared CNN encodes video frames and a bidirectional LSTM models the jump sequence.
- **Group-aware evaluation:** all 12 camera views of each physical jump attempt stay in the same data partition.
- **Controlled experiments:** compare training-only augmentation, AdamW, and learning-rate scheduling against the same baseline.
- **Inspectable results:** per-video predictions, confusion matrices, and class, camera, and skater breakdowns are included in the repository.
- **Experiment traceability:** new training and evaluation runs save source/configuration snapshots, environment metadata, and SHA-256 fingerprints for inputs and results.

## How it works

The browser accepts one MP4 containing a single jump, shows a preview, and sends it to a FastAPI `POST /predict` endpoint. The backend validates the upload, runs inference with a model loaded once at startup, and removes the temporary video after the request. Uploads are limited to 100 MB by default.

OpenCV uniformly samples 32 frames per clip, converts them to 224 x 224 RGB images, and normalizes values to `[0, 1]`. A shared CNN extracts spatial features from each frame; a bidirectional LSTM models their sequence; a classifier produces six output scores. The interface does not find jumps in full programs or score jump quality.

## Data and results

The [FS-Jump3D](https://github.com/ryota-skating/FS-Jump3D) RGB data used here has **2,880 six-class videos from 240 physical jump attempts**. Each attempt was captured from 12 camera views. Train, validation, and test assignments are made by physical attempt, keeping all views of the same jump together to prevent data leakage. Combination jumps are excluded.

| Checkpoint | Validation accuracy / macro F1 | Held-out test accuracy / macro F1 |
| --- | ---: | ---: |
| CNN-BiLSTM baseline | 67.13% / 0.673 | 65.97% / 0.652 |
| CNN-BiLSTM + training-only augmentation | 70.37% / 0.701 | **71.99% / 0.719** |

The augmented model also achieved **83.33% attempt-level accuracy** when averaging predictions across all 12 camera views. This multi-view result is reported separately from single-video classification.

See the [controlled experiment comparison](data/experiments/improvement_phase1/comparison.md) and [held-out test report](data/evaluation/phase1_augmentation_final/final_comparison.md) for the complete results. Detailed predictions and confusion matrices are saved under `data/evaluation/`.

**Evaluation scope:** controlled footage from four skaters, with physical attempts held out across splits. All four skaters appear in each partition; unseen-skater and uncontrolled-video generalization remain future evaluation tasks.

**Web demo:** the current demo uses the baseline checkpoint (65.97% test accuracy), configured in `configs/inference.yaml`. The 71.99% result belongs to the separately preserved augmentation checkpoint.

## Run locally

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn --app-dir src fs_jump3d.web:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**, then upload one already-trimmed single-jump MP4. The model checkpoint is not included in Git: the web app requires the exact checkpoint specified in `configs/inference.yaml` and will fail to start without it. Raw FS-Jump3D videos are also not included.

For command-line prediction with the same checkpoint:

```bash
.venv/bin/python scripts/predict_video.py path/to/trimmed_jump.mp4
```

Run the automated tests with `.venv/bin/python -m pytest`. Model, data, and API code is in `src/fs_jump3d/`; the browser UI is in `web/`, and reproducible configs and CLI scripts are in `configs/` and `scripts/`.

## Reproduce the experiments

The [reproducibility guide](docs/REPRODUCIBILITY.md) covers dataset preparation, training, validation, held-out evaluation, and artifact verification. A pinned snapshot of the current environment is available in [`environments/requirements-current.txt`](environments/requirements-current.txt).

```bash
.venv/bin/python scripts/train.py --config configs/train.yaml --run-dir models/runs/reproduction/baseline
.venv/bin/python scripts/train.py --config configs/improvement_phase1_augmentation.yaml --run-dir models/runs/reproduction/augmentation
```

Each run writes its configuration, training history, best checkpoint, and provenance records to its own directory.
