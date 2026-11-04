import hashlib
import runpy
from pathlib import Path

import pytest
import yaml


def test_install_checks_hash_and_preserves_existing_checkpoint(tmp_path):
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/install_model.py'))
    install = module['install']
    install.__globals__['ROOT'] = tmp_path
    (tmp_path / 'configs').mkdir()
    source = tmp_path / 'source.pt'
    source.write_bytes(b'checkpoint fixture')
    config = {'checkpoint': 'models/best.pt',
              'expected_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    (tmp_path / 'configs/inference.yaml').write_text(yaml.safe_dump(config))
    target = install(source)
    assert target.read_bytes() == source.read_bytes()
    assert install(source) == target
    target.write_bytes(b'another checkpoint')
    with pytest.raises(FileExistsError):
        install(source)
    assert target.read_bytes() == b'another checkpoint'
    source.write_bytes(b'wrong bytes')
    with pytest.raises(ValueError, match='SHA-256'):
        install(source)
