from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from fastapi.testclient import TestClient

from fs_jump3d.dataset import TARGET_CLASSES
from fs_jump3d.inference import CheckpointError, InferenceResult
from fs_jump3d.preprocessing import VideoDecodeError
from fs_jump3d.web import create_app


class FakePredictor:
    def __init__(self, error=None):
        self.paths = []
        self.error = error

    def predict(self, path):
        self.paths.append(Path(path))
        assert Path(path).is_file()
        if self.error:
            raise self.error
        probabilities = dict.fromkeys(TARGET_CLASSES, 0.04)
        probabilities["Lutz"] = 0.8
        return InferenceResult("Lutz", 0.8, probabilities, "fake", "cpu", {}, 0.1, 0.1, 0.2)


def mp4_bytes():
    return b"\x00\x00\x00\x18ftypisom" + b"x" * 20


def test_startup_loads_predictor_once_and_home():
    predictor = FakePredictor()
    calls = []

    def factory():
        calls.append(1)
        return predictor

    with TestClient(create_app(factory)) as client:
        assert client.get("/").status_code == 200
        assert "Analyze a jump" in client.get("/").text
        assert client.get("/app.js").status_code == 200
        for _ in range(2):
            response = client.post("/predict", files={"video": ("clip.mp4", mp4_bytes(), "video/mp4")})
            assert response.status_code == 200
        assert len(calls) == 1
        assert len(predictor.paths) == 2
        assert all(not path.exists() for path in predictor.paths)
        data = response.json()
        assert data["predicted_label"] in TARGET_CLASSES
        assert len(data["class_probabilities"]) == 6
        assert sum(data["class_probabilities"].values()) == pytest.approx(1)


def test_startup_failure_is_clear():
    def broken():
        raise CheckpointError("/private/checkpoint.pt")

    with pytest.raises(RuntimeError, match="Web startup failed"):
        with TestClient(create_app(broken)):
            pass


def test_upload_validation_and_cleanup(monkeypatch):
    predictor = FakePredictor()
    with TestClient(create_app(lambda: predictor)) as client:
        assert client.post("/predict").status_code == 422
        assert client.post("/predict", files={"video": ("bad.txt", b"abc", "video/mp4")}).status_code == 415
        assert client.post("/predict", files={"video": ("empty.mp4", b"", "video/mp4")}).status_code == 400
        assert client.post("/predict", files={"video": ("fake.mp4", b"not a video", "video/mp4")}).status_code == 415
        client.app.state.max_upload_bytes = 10
        with TemporaryDirectory() as temporary:
            monkeypatch.setattr("fs_jump3d.web.NamedTemporaryFile", lambda **kwargs: __import__("tempfile").NamedTemporaryFile(dir=temporary, **kwargs))
            response = client.post("/predict", files={"video": ("large.mp4", mp4_bytes(), "video/mp4")})
            assert response.status_code == 413
            assert list(Path(temporary).iterdir()) == []


def test_decode_failure_hides_paths_and_cleans_up():
    predictor = FakePredictor(VideoDecodeError("/private/secret.mp4 unreadable"))
    with TestClient(create_app(lambda: predictor)) as client:
        response = client.post("/predict", files={"video": ("clip.mp4", mp4_bytes(), "application/octet-stream")})
        assert response.status_code == 422
        assert "/private/" not in response.text
        assert not predictor.paths[0].exists()


def test_prediction_failure_is_readable():
    predictor = FakePredictor(RuntimeError("/private/secret.mp4"))
    with TestClient(create_app(lambda: predictor)) as client:
        response = client.post("/predict", files={"video": ("clip.mp4", mp4_bytes(), "video/mp4")})
        assert response.status_code == 500
        assert "/private/" not in response.text
        assert not predictor.paths[0].exists()
