import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from .labels import CLASS_NAMES
from .preprocessing import DEFAULT_IMAGE_SIZE, DEFAULT_NUM_FRAMES, sample_video_frames


def predict_video(
    video_path: str | Path,
    model_path: str | Path = "models/jump_classifier.keras",
    num_frames: int = DEFAULT_NUM_FRAMES,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> dict:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first with: python -m src.jump_classifier.train"
        )

    model = load_model(model_path)
    frames = sample_video_frames(video_path, num_frames, image_size)
    probabilities = model.predict(np.expand_dims(frames, axis=0), verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    return {
        "label": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            label: float(probabilities[index]) for index, label in enumerate(CLASS_NAMES)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one pre-cropped jump video.")
    parser.add_argument("video_path")
    parser.add_argument("--model-path", default="models/jump_classifier.keras")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE[0])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict_video(
        args.video_path,
        args.model_path,
        args.num_frames,
        (args.image_size, args.image_size),
    )
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
