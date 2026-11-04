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
- Tests cover unchanged optimization with/without record collection, missing
  experiment seals, environment mismatch, stale/tampered cache detection, legacy
  cache validation, and hash-checked checkpoint installation.

The original environment, full split, model weights and scientific evaluation
reports were not regenerated. A dated deployment note was added to the historical
report. The GitHub Actions workflow is configured but has not yet been observed
running on GitHub. Browser rendering was not rechecked during this maintenance.

Remaining research work: repeat-seed experiments and, if the research question
requires it, unseen-skater evaluation. Public model distribution remains pending;
see `DATA_AND_MODELS.md`. These are separate from the engineering checks above.
