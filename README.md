# Figure Skating Jump Recognition

End-to-end PyTorch project for classifying already-trimmed single-jump figure skating MP4 clips into six jump types: Axel, Flip, Loop, Lutz, Salchow, and Toeloop.

Milestones 1-7 are complete: audit, leakage-aware split, OpenCV preprocessing, CNN-BiLSTM training, held-out evaluation, single-video inference, and a local web MVP.

## Run the Web MVP

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn --app-dir src fs_jump3d.web:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. The ignored fixed checkpoint `models/runs/milestone4/best_model.pt` must be present and match the SHA-256 in `configs/inference.yaml`; startup fails clearly otherwise. Select or drop one already-trimmed single-jump MP4, preview it, then click **Analyze jump**. The page displays the predicted class, model confidence, and all six probabilities. `POST /predict` accepts multipart field `video` and returns structured JSON. The 100 MB default upload limit is set in `configs/web.yaml` and can be overridden with `FS_JUMP3D_MAX_UPLOAD_MB`. Uploads use generated files in the system temporary directory and are deleted after each request.

The model path is OpenCV frame sampling -> shared custom CNN frame encoder -> BiLSTM -> six-class classifier. The six classes are Axel, Flip, Loop, Lutz, Salchow, and Toeloop. The held-out grouped test result is 65.97% video-level accuracy and 0.652 macro F1. This is a four-skater controlled dataset; confidence is not guaranteed correctness, and arbitrary broadcast/phone/internet video performance is not established. The MVP does not detect jumps in full programs, classify combinations, count rotations, or score GOE/quality.

Main project directories: `src/fs_jump3d/` (model, preprocessing, inference, web API), `web/` (static interface), `configs/` (reproducible settings), `scripts/` (CLI workflows), `tests/`, `data/audit/`, `data/splits/`, and ignored `data/raw/`, `data/processed/`, `models/`.

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

## Milestone 3 Preprocessing

The deterministic preprocessing pipeline converts one MP4 clip into a `float32` tensor with shape `[32, 3, 224, 224]`.

It uses OpenCV to decode frames, samples 32 frames uniformly across the full clip, converts BGR to RGB, resizes to `224x224`, and normalizes pixel values to `[0, 1]`. If a video has fewer than 32 decodable frames, all decoded frames are used in order and the final valid frame is repeated.

Run decode/preprocessing validation with:

```bash
python3 scripts/validate_preprocessing.py --split-csv data/splits/dataset_split.csv --data-root data/raw/fs-jump3d --output-dir data/preprocessing
```

Milestone 3 validation opened and decoded the first frame for all 2,880 six-class split videos. Representative clips were fully preprocessed to verify tensor shape, dtype, range, and sampling indices.

## Milestone 4 Training

The PyTorch Dataset reads `data/splits/dataset_split.csv`. Training uses only `train` rows and validation uses only `val` rows. The `test` rows are not loaded by the training DataLoaders. The fixed label order is Axel, Flip, Loop, Lutz, Salchow, Toeloop.

The model applies one shared custom CNN to all 32 frames, projects each frame to 128 features, then passes the sequence to a one-layer bidirectional LSTM with 128 hidden units per direction. A dropout/dense head returns six raw logits. Training uses CrossEntropyLoss and Adam.

Run the smoke check and full training from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/train.py --smoke
.venv/bin/python scripts/train.py
```

The initial settings are in `configs/train.yaml`: seed 42, 32 frames, 224-pixel input, batch size 4, at most 30 epochs, learning rate 1e-4, early stopping patience 5, and two DataLoader workers. The run uses MPS when available, otherwise CUDA or CPU. Training streams decoded frames and retains only the selected 32, matching the Milestone 3 output without holding all 300 full-resolution frames in memory. Deterministic preprocessed frame caches under `data/processed/` and checkpoints under `models/` are ignored by Git. Each run writes a resolved config, epoch history, summary, and best checkpoint. Validation loss selects the checkpoint; no test metrics are computed here.

The first run trained for 24 epochs on MPS and stopped early. Epoch 19 produced the best validation loss, 0.7804, with 67.13% validation accuracy. The 395,222-parameter model and run details are documented in `data/training/milestone4/summary.md`; the checkpoint is at `models/runs/milestone4/best_model.pt` and is not tracked. Validation metrics varied markedly across epochs, so use the saved best checkpoint and consult the full history rather than the last epoch alone.

## Milestone 5 Evaluation

The fixed epoch 19 checkpoint was evaluated once on the 432-video, 36-attempt grouped test split. Video-level accuracy is **65.97%** and macro F1 is **0.652**. Averaging class probabilities across each attempt's 12 camera views gives **72.22%** group-level accuracy and **0.704** group-level macro F1. These are different units of analysis.

The full report, per-video predictions, per-group predictions, metrics, confusion matrices, and plots are in `data/evaluation/milestone5/`. The strongest video-level class by F1 is Axel; Toeloop is weakest, with 43 of its 72 views predicted as Salchow. Camera accuracy varies from 47.22% to 80.56%. All four skaters appear in the train, validation, and test splits, so this is not unseen-skater generalization.

To verify the checkpoint and test split without evaluating videos, run:

```bash
.venv/bin/python scripts/evaluate.py --verify-only
```

`scripts/evaluate.py` reproduces the held-out evaluation artifacts. The test result is for the fixed Milestone 4 model; do not use it to select a different checkpoint or tune this model.

## Milestone 6 Single-Video Inference

Classify one already-trimmed MP4 containing one primary jump:

```bash
.venv/bin/python scripts/predict_video.py path/to/trimmed_jump.mp4
```

The CLI prints the predicted class, confidence, all six class probabilities, device, and timing. `src/fs_jump3d/inference.py` provides a reusable `VideoPredictor.from_config()` and structured `InferenceResult` for the later web layer. `configs/inference.yaml` pins the evaluated epoch 19 checkpoint by SHA-256. The model loads on MPS when available, with CPU fallback, and uses the same deterministic OpenCV preprocessing as training/evaluation. It needs only one clip, not 12 camera views.

Engineering validation on three train/validation clips is recorded in `data/inference/milestone6/validation_summary.md`. Warmed inference was about one second per clip, mostly OpenCV decoding; the web app should load the predictor once at startup. This model does not detect or trim jumps in full programs, classify combinations, count rotations, score quality/GOE, or guarantee generalization to arbitrary broadcast, phone, or internet footage. Softmax confidence is not a calibrated guarantee.

## Tests

Run tests with:

```bash
.venv/bin/python -m pytest
```
