# Figure Skating Single-Jump Classifier MVP

This project classifies a pre-cropped single-jump skating clip into one of six jump types:

- Axel
- Lutz
- Flip
- Loop
- Salchow
- Toe Loop

The MVP is intentionally simple:

1. Upload one already-trimmed MP4 clip containing one jump.
2. Use OpenCV to sample frames from the video.
3. Resize and normalize the frames.
4. Train a CNN-BiLSTM classifier from scratch.
5. Evaluate with accuracy, precision, recall, F1, and a confusion matrix.
6. Run inference from a simple web upload page.
7. Use pretrained YOLO to draw a skater/person bounding box overlay on the uploaded video.

This project does not do full-video jump detection, LLMs, RAG, agents, or MCP.

## Dataset Layout

Put your labeled clips here:

```text
data/raw/
  Axel/
    axel_001.mp4
    axel_002.mp4
  Lutz/
    lutz_001.mp4
  Flip/
  Loop/
  Salchow/
  Toe Loop/
```

The folder name is the label. Each video should already contain only one jump.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.jump_classifier.train --data-dir data/raw --epochs 20
```

The trained model is saved to:

```text
models/jump_classifier.keras
```

Reports are saved to:

```text
reports/
```

## Predict One Video

```bash
python -m src.jump_classifier.predict path/to/jump.mp4
```

## Run Web App

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Upload a pre-cropped MP4 and the page will show:

- the original video
- a YOLO annotated video with the skater/person bounding box
- the predicted jump type
- confidence scores

## YOLO Overlay

YOLO is part of the MVP. The app uses pretrained YOLO to detect the person/skater in each frame and write an annotated video.

YOLO is only for drawing the bounding box. The jump type still comes from the CNN-BiLSTM model.
