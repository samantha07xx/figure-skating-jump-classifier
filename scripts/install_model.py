#!/usr/bin/env python3
"""Verify and install the bundled evaluated checkpoint (or a supplied copy)."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]


def install(source: Path) -> Path:
    config = yaml.safe_load((ROOT / 'configs/inference.yaml').read_text())
    expected = config['expected_sha256']
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError('Checkpoint SHA-256 does not match the evaluated model; nothing installed')
    target = ROOT / config['checkpoint']
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return target
        raise FileExistsError(f'Refusing to replace a different checkpoint: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.pt', delete=False) as stream:
        temporary = Path(stream.name)
        with source.open('rb') as origin:
            shutil.copyfileobj(origin, stream)
    try:
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            raise ValueError('Source changed during copy; nothing installed')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, nargs='?',
                        default=ROOT / 'artifacts/augmentation-v1/best_model.pt',
                        help='Optional checkpoint path; defaults to the bundled evaluated model')
    args = parser.parse_args()
    print(install(args.checkpoint))
