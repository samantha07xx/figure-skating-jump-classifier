# Figure Skating Jump Recognition

A local web app that classifies an already-trimmed figure-skating jump video as **Axel, Flip, Loop, Lutz, Salchow, or Toe Loop**. It uses OpenCV video preprocessing and a custom PyTorch CNN-BiLSTM, then displays the predicted class, model confidence, and probabilities for all six classes. The web-served model achieved **65.97% video-level accuracy** on the held-out FS-Jump3D test split.

## How it works

The browser accepts one MP4 containing a single jump, shows a preview, and sends it to a FastAPI `POST /predict` endpoint. The backend validates the upload, runs inference with a model loaded once at startup, and removes the temporary video after the request. Uploads are limited to 100 MB by default.

OpenCV uniformly samples 32 frames per clip, converts them to 224 x 224 RGB images, and normalizes values to `[0, 1]`. A shared CNN extracts spatial features from each frame; a bidirectional LSTM models their sequence; a classifier produces six output scores. The interface does not find jumps in full programs or score jump quality.

## Data and results

The [FS-Jump3D](https://github.com/ryota-skating/FS-Jump3D) RGB data used here has **2,880 six-class videos from 240 physical jump attempts**. Each attempt was captured from 12 camera views. Train, validation, and test assignments are made by physical attempt, keeping all views of the same jump together to prevent data leakage. Combination jumps are excluded.

| Checkpoint | Validation accuracy / macro F1 | Held-out test accuracy / macro F1 |
| --- | ---: | ---: |
| Web-served model | 67.13% / 0.673 | 65.97% / 0.652 |
| Separately evaluated model with training-only augmentation | 70.37% / 0.701 | 71.99% / 0.719 |

The augmented checkpoint is **not** used by the web app. Detailed per-class metrics and confusion matrices are saved under `data/evaluation/`. Results are for controlled footage from four skaters and do not establish performance on unseen skaters or arbitrary broadcast, phone, or internet videos.

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
