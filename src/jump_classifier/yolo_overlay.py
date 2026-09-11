from pathlib import Path

import cv2


def create_yolo_overlay(
    input_video_path: str | Path,
    output_video_path: str | Path,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.25,
) -> Path:
    """Draw pretrained YOLO person/skater boxes on a video."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLO overlay requires ultralytics. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(model_name)
    capture = cv2.VideoCapture(str(input_video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 24
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError(f"Could not read video dimensions: {input_video_path}")

        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError(f"Could not create annotated video: {output_video_path}")

        while True:
            success, frame = capture.read()
            if not success:
                break

            # COCO class 0 is "person"; for this MVP, the skater is treated as the person in frame.
            results = yolo.predict(
                frame,
                classes=[0],
                conf=confidence_threshold,
                verbose=False,
            )
            annotated = results[0].plot()
            writer.write(annotated)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return output_video_path
