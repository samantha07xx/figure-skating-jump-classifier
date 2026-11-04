# Reproducing research runs

This repository separates reproducible experiment bookkeeping from claims about
model quality. The organization update preserves the CNN-BiLSTM architecture,
initialization, random seeding, data order, augmentation, preprocessing, loss,
optimizer, scheduler, checkpoint selection, inference API and web interface.
New bookkeeping validates splits, records input bytes, and rejects overwriting
existing runs. It does not retrain or replace historical checkpoints.

## Environment

Run commands from the repository root. The existing `.venv` remains untouched.
`requirements.txt` describes supported dependency lower bounds;
`environments/requirements-current.txt` pins the installed packages observed during
this organization update. Its header records Python and platform. It is an
environment snapshot, **not evidence of the original training environment** and
not a tested cross-platform dependency lock. Package availability on a fresh
machine has not been verified. Preserve wheel files separately for long-term
offline reproduction.

To recreate that snapshot on a compatible Python/platform in a new environment:

```bash
python3.13 -m venv .venv-reproduction
.venv-reproduction/bin/python -m pip install -r environments/requirements-current.txt
.venv-reproduction/bin/python -m pytest -q
```

The commands below use `.venv/bin/python`; substitute the reproduction environment
when appropriate. Prefer the original recorded device. `auto` is retained for
compatibility and can select different devices on different computers.

## Data and fixed protocol

Obtain the RGB videos from the dataset source identified in
`configs/dataset.yaml` and place them under:

```text
data/raw/fs-jump3d/skater_A/cam_1/Axel_1.mp4
```

The dataset is not distributed with this repository. The canonical labels, in
output-index order, are `Axel, Flip, Loop, Lutz, Salchow, Toeloop`. Combination
jumps are excluded. A physical attempt is identified by skater, jump type and
attempt number; all 12 views stay together. The committed split is the reference:
2,016/432/432 videos and 168/36/36 attempts in train/validation/test.

All four skaters occur in all three partitions. This protocol measures held-out
attempt recognition, not unseen-skater generalization. Video metrics and
12-camera probability-averaged metrics are distinct tasks. Camera views of one
attempt are correlated; statistical resampling must respect that grouping.

Inspect/recreate supporting artifacts in new directories, without replacing the
reference artifacts:

```bash
.venv/bin/python scripts/audit_dataset.py --output-dir data/reproduction/audit
.venv/bin/python scripts/create_split.py --index-path data/reproduction/audit/dataset_index.csv --output-dir data/reproduction/splits --seed 42 --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
.venv/bin/python scripts/validate_preprocessing.py --output-dir data/reproduction/preprocessing --full-decode
```

Compare the regenerated `dataset_split.csv` with the committed reference before
using it. Audit/size metadata may differ if the supplied dataset differs.
`configs/dataset.yaml`, `split.yaml` and `preprocess.yaml` document the protocol;
those three scripts take their settings from CLI arguments/defaults, **not**
automatically from these YAML files. Training YAML files are executable settings.
The supported loss remains CrossEntropyLoss; an unsupported `loss` now raises an
error instead of silently being ignored.

## Training

Choose a new run directory for every invocation. An occupied directory is rejected
even if an earlier run was interrupted. This project does not implement exact
resume; checkpoints are selected-model artifacts, not complete resumable states.

```bash
# Tiny pipeline check; still fingerprints the configured train/validation inputs.
.venv/bin/python scripts/train.py --config configs/train.yaml --run-dir models/runs/reproduction/baseline-check --device cpu --smoke

# Full baseline; use the device recorded in the experiment being reproduced.
.venv/bin/python scripts/train.py --config configs/train.yaml --run-dir models/runs/reproduction/baseline --device mps

# Existing augmentation experiment, without changing its scientific settings.
.venv/bin/python scripts/train.py --config configs/improvement_phase1_augmentation.yaml --run-dir models/runs/reproduction/augmentation --device mps
```

The other existing candidate configs are `improvement_phase1_adamw.yaml`,
`improvement_phase1_plateau.yaml` and `improvement_phase1_aug_plateau.yaml`.
Checkpoints are chosen by minimum validation loss, not peak accuracy. The
30-epoch cap and early stopping behavior are unchanged. Changing the seed is a
new replicate, not reproduction of seed 42; save the edited YAML with the run.

Every new training run contains:

```text
resolved_config.yaml
history.csv
summary.json
best_model.pt
provenance/
  manifest.json       # git status, package/device/backend state, config, SHA-256 inputs
  outputs.json        # result hashes bound to the input manifest hash
  split.csv          # exact split bytes, retaining original row order
  source/            # source, scripts, configs, tests, UI, docs and environment snapshot
```

Fingerprints cover raw videos in the consumed partitions and existing frame
caches, with missing caches recorded explicitly. Hashing adds startup I/O but
does not alter any tensors, RNG states or backend settings. A run must have
exclusive access to its output directory; keep code, source data and caches
unchanged while it runs. Do not run multiple writers against the same cache.

### Cache and numerical behavior

For compatibility, the existing uint8 frame cache format and lookup behavior are
preserved. A manifest identifies the exact cache files used; it does not certify
that an old cache was generated from the currently supplied raw videos.
If videos or preprocessing change, select a **new `cache_dir` in a copied training
config**. Keep the original cache for historical reproduction. Do not silently
reuse it for a different dataset.

To reproduce a recorded warm-cache run, retain the matching recorded caches.
For a cold-cache run, use a fresh cache directory and consult
`missing_caches_at_start`. Existing files are hashed; absent files cannot be
verified after generation as if they had existed initially. Hashing adds no
cache invalidation or decoding changes.

Seeds are fixed as before. Deterministic backend algorithms are not forcibly
enabled because that can change supported operations, performance and numerical
behavior. Backend flags are recorded. Exact bitwise agreement across OS, GPU,
PyTorch/OpenCV versions or hardware is **not promised**. Even on matching hardware,
nondeterministic kernels can prevent bitwise reproduction. Report that limitation
and use repeated-seed experiments for stability claims. The current historical
experiments use one seed and do not establish statistical significance.

## Validation and test evaluation

Evaluate validation candidates first. Every evaluation records source/environment,
checkpoint and dataset fingerprints. Validation also records the frame cache.
Output paths must be new.

```bash
.venv/bin/python scripts/evaluate_validation.py models/runs/reproduction/baseline/best_model.pt data/reproduction/baseline_validation.json --device mps
```

For test evaluation, read `best_epoch` from the new run's `summary.json` and pass it
as `--expected-epoch`; it need not match a historical epoch. For example, **only
if the new run selected epoch 19**:

```bash
.venv/bin/python scripts/evaluate.py --checkpoint models/runs/reproduction/baseline/best_model.pt --training-summary models/runs/reproduction/baseline/summary.json --expected-epoch 19 --output-dir data/reproduction/baseline_test --device mps
```

The current test evaluator intentionally retains the reference protocol checks:
432 videos, 36 complete groups, six classes, 32 frames and 224-pixel inputs.
It is not a generic evaluator for arbitrary new split protocols.
`--verify-only` checks the checkpoint and split without executing predictions or
creating a provenance run. The validation output's companion directory is named
`<output filename>.provenance`.

Test results are for final reporting. Do not repeatedly select methods on them.
Historical test artifacts remain under `data/evaluation/`; existing model hashes
and reported scores are not rewritten by this update.

## Verify or archive

```bash
# Capture the current dataset/cache/checkpoint state without training.
.venv/bin/python scripts/reproducibility.py capture --output-dir data/reproduction/current-snapshot --checkpoint models/runs/milestone4/best_model.pt

# Verify source, data, caches, source/split snapshots and sealed results.
.venv/bin/python scripts/reproducibility.py verify data/reproduction/current-snapshot/manifest.json
.venv/bin/python scripts/reproducibility.py verify models/runs/reproduction/baseline/provenance/manifest.json
```

Verification exits nonzero and lists changed/missing files. A later code edit
correctly causes the source comparison to fail; restore/use the archived `source/`
when reproducing that run. `--root` relocates project-relative records to a new
checkout. Inputs outside the repository retain absolute paths and must be restored
there or explicitly remapped in a separate copy of the manifest; editing a sealed
manifest intentionally invalidates its seal. SHA-256 records detect drift, not
malicious tampering with both files and records.

Archive the entire run plus its matching raw data and caches using suitable
storage; these large files are intentionally excluded from Git. Commit small
configs and research summaries. Capturing an old checkpoint today establishes
its present identity only: it cannot recover the original code/environment or
prove historical execution order.

## Web application

```bash
.venv/bin/uvicorn --app-dir src fs_jump3d.web:app --host 127.0.0.1 --port 8000
```

The web demo now uses the evaluated augmentation checkpoint (epoch 30), pinned
by SHA-256 in `configs/inference.yaml`. The baseline remains available separately.
New research runs do not automatically replace the deployed model.
