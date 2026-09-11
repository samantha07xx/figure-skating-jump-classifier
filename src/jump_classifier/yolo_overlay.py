from pathlib import Path

import cv2


def create_yolo_overlay(
    input_video_path: str | Path,
    output_video_path: str | Path,
    model_name: str = "yolov8n.pt",
) -> Path:
    """Optional helper: draw pretrained YOLO person boxes on a video."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLO overlay requires ultralytics. Install it with: pip install ultralytics"
        ) from exc

    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(model_name)
    capture = cv2.VideoCapture(str(input_video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    while True:
        success, frame = capture.read()
        if not success:
            break

        results = yolo(frame, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)

    capture.release()
    writer.release()
    return output_video_path
