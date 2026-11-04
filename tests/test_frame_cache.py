import json

import numpy as np
import pytest

from fs_jump3d import preprocessing
from fs_jump3d.frame_cache import cached_frames


def test_cache_detects_input_pipeline_and_cache_changes(tmp_path, monkeypatch):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'video-v1')
    cache = tmp_path / 'cache' / 'video.npy'
    config = preprocessing.PreprocessConfig(2, 4)
    frames = np.full((2, 3, 4, 4), 128 / 255, dtype=np.float32)
    calls = []
    def decode(*args):
        calls.append(1)
        return type('Decoded', (), {'frames': frames})()
    monkeypatch.setattr(preprocessing, 'preprocess_video_streaming', decode)
    np.testing.assert_array_equal(cached_frames(video, cache, config), frames)
    np.testing.assert_array_equal(cached_frames(video, cache, config), frames)
    assert len(calls) == 1
    video.write_bytes(b'video-v2')
    with pytest.raises(ValueError, match='stale'):
        cached_frames(video, cache, config)
    video.write_bytes(b'video-v1')
    metadata = cache.with_suffix('.npy.json')
    original = metadata.read_text()
    changed = json.loads(original)
    changed['preprocessing_sha256'] = 'other-version'
    metadata.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='stale'):
        cached_frames(video, cache, config)
    metadata.write_text(original)
    np.save(cache, np.zeros_like(frames, dtype=np.uint8))
    with pytest.raises(ValueError, match='stale'):
        cached_frames(video, cache, config)


def test_legacy_cache_requires_exact_tensor_match(tmp_path, monkeypatch):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'video')
    cache = tmp_path / 'video.npy'
    np.save(cache, np.zeros((2, 3, 4, 4), dtype=np.uint8))
    config = preprocessing.PreprocessConfig(2, 4)
    frames = np.ones((2, 3, 4, 4), dtype=np.float32)
    monkeypatch.setattr(preprocessing, 'preprocess_video_streaming',
                        lambda *args: type('Decoded', (), {'frames': frames})())
    with pytest.raises(ValueError, match='legacy'):
        cached_frames(video, cache, config)
    assert not cache.with_suffix('.npy.json').exists()
    frames[:] = 0
    np.testing.assert_array_equal(cached_frames(video, cache, config), frames)
    assert cache.with_suffix('.npy.json').is_file()
