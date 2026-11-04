"""Validate cache provenance without changing the existing uint8 tensor format."""
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from fs_jump3d import preprocessing
from fs_jump3d.reproducibility import sha256_file


def cached_frames(video: Path, cache: Path, config: preprocessing.PreprocessConfig) -> np.ndarray:
    identity = {
        "schema_version": 1,
        "video_sha256": sha256_file(video),
        "preprocessing_sha256": sha256_file(Path(preprocessing.__file__)),
        "cache_code_sha256": sha256_file(Path(__file__)),
        "config": asdict(config), "opencv": cv2.__version__, "numpy": np.__version__,
    }
    metadata = cache.with_suffix(".npy.json")
    expected_shape = (config.frames_per_clip, 3, config.image_size, config.image_size)
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)
        if stored.dtype != np.uint8 or stored.shape != expected_shape:
            raise ValueError(f"invalid frame cache: {cache}; use a new cache_dir")
        frames = stored.astype(np.float32) / 255.0
        identity["cache_sha256"] = sha256_file(cache)
        if metadata.exists():
            if json.loads(metadata.read_text()) != identity:
                raise ValueError(f"stale or modified frame cache: {cache}; use a new cache_dir")
            return frames
        # Legacy caches are adopted only after checking the actual decoded tensor.
        decoded = preprocessing.preprocess_video_streaming(video, config).frames
        if not np.array_equal(frames, decoded):
            raise ValueError(f"legacy cache differs from video: {cache}; use a new cache_dir")
    else:
        frames = preprocessing.preprocess_video_streaming(video, config).frames
        cache.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=cache.parent, suffix=".npy", delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, np.rint(frames * 255.0).astype(np.uint8))
        temporary.replace(cache)
        identity["cache_sha256"] = sha256_file(cache)
    with NamedTemporaryFile(mode="w", dir=cache.parent, suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(identity, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(metadata)
    return frames
