# Repository validation — 2026-11-04

Scope: engineering checks for the research repository. These checks do not add
new model-quality results or replace the saved held-out evaluation.

- Created a separate virtual environment using Python 3.13 on macOS arm64.
- Installed every pinned dependency from `environments/requirements-current.txt`.
- `pip check`: no broken requirements.
- Full test suite: 50 passed. Two upstream Starlette/httpx/AnyIO deprecation warnings remain.
- A real-video CPU smoke run completed one training/validation epoch and saved its
  checkpoint, history, input snapshots, and sealed output fingerprints. It used
  one existing video from each partition in a temporary split, an isolated empty
  cache, and a temporary run directory. Only train/validation videos were consumed.
  The test video was not evaluated; smoke metrics are not scientific results.
- The smoke run's provenance verified successfully with environment checking enabled.
- Bundled-model installation was tested in an isolated checkout copy without a
  pre-existing models directory or raw dataset. The default installer verified and
  installed the exact checkpoint. A real-model API check then accepted an externally
  supplied test video and returned model ID `epoch-30-8ab4aaa77dc4`.
- Tests cover unchanged optimization with/without record collection, missing
  experiment seals, environment mismatch, stale/tampered cache detection, legacy
  cache validation, and hash-checked checkpoint installation.

The original environment, full split, model weights and scientific evaluation
reports were not regenerated. A dated deployment note was added to the historical
report. GitHub Actions passed for commit `1ddd247` (run [41871900641](https://github.com/samantha07xx/figure-skating-jump-classifier/actions/runs/41871900641)); the two preceding runs also passed. Browser rendering was not rechecked during this maintenance.

Remaining research work: repeat-seed experiments and, if the research question
requires it, unseen-skater evaluation. The evaluated model is now bundled for installation; see `DATA_AND_MODELS.md`. These are separate from the engineering checks above.


## Fresh GitHub clone and HTTP deployment

A new clone of commit `ad363cb` was created in a temporary directory with its own
Python environment and no raw dataset or prior run outputs. Installing the
original minimum-version requirements selected PyTorch 2.15.0. Although the
unit tests passed, the first MPS HTTP inference aborted with a native
`MPSNDArray initWithBuffer:offset:descriptor:` exception on this Mac.

The default `requirements.txt` now delegates to the already validated pinned
snapshot (PyTorch 2.14.0). After reinstalling those versions in the independent
clone, dependency checks and all 50 Python tests passed. The two JavaScript
request tests also passed. The model installer used only the bundled weights.
The app was started independently on port 8001, leaving the original port-8000
service running.

The homepage, CSS, JavaScript, health and configuration endpoints returned 200.
Twelve external test-video uploads (lexicographically first two test paths per
class) all completed successfully, used model ID `epoch-30-8ab4aaa77dc4`, and
matched all 12 historical predicted labels. Six predictions were correct;
this diagnostic subset is not an estimate of full-test accuracy. A non-MP4
upload returned 415. Videos were supplied from the existing local dataset;
the upstream dataset download was not re-executed.

This was a real local HTTP deployment check on macOS arm64, not cloud hosting,
a browser visual check, full-test-set inference, or a new full training run.
The dependency correction and README quick-start changes were tested locally
and need their own commit/CI run after this record is updated.
