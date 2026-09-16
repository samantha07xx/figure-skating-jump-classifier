from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
import yaml

from fs_jump3d.inference import CheckpointError, InferenceInputError, VideoPredictor
from fs_jump3d.preprocessing import VideoDecodeError


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "web"
CHUNK_SIZE = 1024 * 1024


def load_upload_limit() -> int:
    config = yaml.safe_load((ROOT / "configs/web.yaml").read_text())
    value = int(os.environ.get("FS_JUMP3D_MAX_UPLOAD_MB", config["max_upload_mb"]))
    if value < 1:
        raise ValueError("max_upload_mb must be positive")
    return value * 1024 * 1024


def create_app(predictor_factory=VideoPredictor.from_config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.predictor = predictor_factory()
            app.state.max_upload_bytes = load_upload_limit()
            app.state.predict_lock = Lock()
        except (CheckpointError, OSError, ValueError) as exc:
            raise RuntimeError("Web startup failed: check the pinned model checkpoint and web configuration") from exc
        yield

    app = FastAPI(title="Figure Skating Jump Recognition", lifespan=lifespan)

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/app.css")
    def css():
        return FileResponse(STATIC / "app.css", media_type="text/css")

    @app.get("/app.js")
    def js():
        return FileResponse(STATIC / "app.js", media_type="text/javascript")

    @app.get("/health")
    def health():
        return {"status": "ready"}

    @app.get("/config")
    def config():
        return {"max_upload_bytes": app.state.max_upload_bytes}

    @app.post("/predict")
    async def predict(video: UploadFile = File(...)):
        if not video.filename or Path(video.filename).suffix.lower() != ".mp4":
            raise HTTPException(415, "Upload one MP4 video.")
        path = None
        try:
            with NamedTemporaryFile(suffix=".mp4", prefix="fs-jump3d-", delete=False) as temp:
                path = Path(temp.name)
                size = 0
                header = b""
                while chunk := await video.read(CHUNK_SIZE):
                    size += len(chunk)
                    if size > app.state.max_upload_bytes:
                        raise HTTPException(413, "Video exceeds the upload size limit.")
                    if len(header) < 12:
                        header += chunk[:12 - len(header)]
                    temp.write(chunk)
            if size == 0:
                raise HTTPException(400, "The uploaded video is empty.")
            if len(header) < 12 or header[4:8] != b"ftyp":
                raise HTTPException(415, "This file is not a readable MP4 video.")

            def run_prediction():
                with app.state.predict_lock:
                    return app.state.predictor.predict(path)

            result = await run_in_threadpool(run_prediction)
            return result.to_dict()
        except (InferenceInputError, VideoDecodeError) as exc:
            raise HTTPException(422, "The video could not be decoded. Upload a playable single-jump MP4.") from exc
        except CheckpointError as exc:
            raise HTTPException(503, "Prediction is temporarily unavailable.") from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, "Prediction failed. Please try another MP4.") from exc
        finally:
            await video.close()
            if path is not None:
                path.unlink(missing_ok=True)

    return app


app = create_app()
