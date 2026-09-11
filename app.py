from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from src.jump_classifier.predict import predict_video

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "models" / "jump_classifier.keras"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

app = Flask(__name__)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def is_allowed_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    video_url = None

    if request.method == "POST":
        uploaded_file = request.files.get("video")
        if not uploaded_file or uploaded_file.filename == "":
            error = "Please choose a video file."
        elif not is_allowed_video(uploaded_file.filename):
            error = "Please upload an MP4, MOV, AVI, or MKV video."
        else:
            original_name = secure_filename(uploaded_file.filename)
            saved_name = f"{uuid4().hex}_{original_name}"
            saved_path = UPLOAD_DIR / saved_name
            uploaded_file.save(saved_path)
            video_url = f"/uploads/{saved_name}"

            try:
                result = predict_video(saved_path, MODEL_PATH)
            except Exception as exc:
                error = str(exc)

    return render_template("index.html", result=result, error=error, video_url=video_url)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
