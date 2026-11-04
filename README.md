# Figure Skating Jump Recognition

CNN-BiLSTM classification of six figure-skating jumps from trimmed RGB video clips: Axel, Flip, Loop, Lutz, Salchow, and Toe Loop. This repository contains data preparation, training, evaluation, and single-video inference code, with a FastAPI web demo.

## Method

OpenCV uniformly samples 32 frames per clip, resizes them to 224 × 224, converts them to RGB, and normalizes pixel values to `[0, 1]`. A shared CNN encodes each frame; a bidirectional LSTM models the sequence; a classifier outputs scores for the six jump classes.

Experiments compare the baseline with clip-consistent spatial and photometric augmentation, AdamW, and learning-rate scheduling. Augmentation is applied only during training. The task assumes an already-trimmed single jump; jump detection in full programs and jump-quality scoring are outside its scope.

## Dataset and experimental protocol

The six-class subset of [FS-Jump3D](https://github.com/ryota-skating/FS-Jump3D) used here contains 2,880 videos from 240 physical jump attempts by four skaters, with 12 camera views per attempt. Combination jumps are excluded.

Data are split by physical attempt, keeping all camera views of an attempt in one partition:

| Partition | Physical attempts | Videos |
| --- | ---: | ---: |
| Training | 168 | 2,016 |
| Validation | 36 | 432 |
| Test | 36 | 432 |

Experiments use seed 42, the same split and model architecture, a 30-epoch limit, and early stopping with patience 5. Each run's checkpoint is selected by minimum validation loss. Candidate methods are compared on validation data before the selected augmentation model is evaluated on the held-out test set.

All four skaters appear in each partition. This protocol evaluates held-out attempts under the dataset's recording conditions; it does not measure unseen-skater or uncontrolled-video generalization. Results are from single-seed runs.

## Results

Accuracy and macro F1 below are computed per video, treating each camera view as one prediction.

| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: | ---: |
| CNN-BiLSTM baseline | 67.13% | 0.673 | 65.97% | 0.652 |
| CNN-BiLSTM + training-only augmentation | 70.37% | 0.701 | 71.99% | 0.719 |

The [experiment comparison](data/experiments/improvement_phase1/comparison.md) reports all candidate runs and validation-based selection. The [test report](data/evaluation/phase1_augmentation_final/final_comparison.md) includes per-class metrics, camera and skater breakdowns, error analysis, and a separate multi-view evaluation that averages probabilities across an attempt's camera views. Prediction files and confusion matrices are available under `data/evaluation/`.

## Quick start: run the demo

Use Python 3.13. The following commands were verified on macOS arm64. The default
requirements install the pinned dependency set used by the validated environment
and CI. A trained checkpoint is included; no retraining or full dataset download
is needed to launch the demo.

```bash
git clone https://github.com/samantha07xx/figure-skating-jump-classifier.git
cd figure-skating-jump-classifier
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/install_model.py
.venv/bin/uvicorn --app-dir src fs_jump3d.web:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and upload a trimmed single-jump MP4. Supply your own
clip or obtain videos from the [dataset source](docs/DATA_AND_MODELS.md); raw
videos are not bundled. The interface displays the predicted class and softmax
probabilities. The default upload limit is 100 MB. Stop the server with Ctrl+C.
If port 8000 is occupied, choose another port such as `--port 8001`.

The installer verifies the bundled checkpoint's SHA-256 before placing it at the
path in `configs/inference.yaml`. Other training outputs remain excluded from Git.
For command-line prediction with the same model:

```bash
.venv/bin/python scripts/predict_video.py path/to/trimmed_jump.mp4
```

Optional checks: `.venv/bin/python -m pytest` runs the Python tests;
`node --test tests/web_requests.test.cjs` checks asynchronous UI request handling
(Node.js 22). Node.js is not required to run the demo. Other operating systems
have not been deployment-tested; Windows uses `.venv\Scripts\python.exe` in
place of `.venv/bin/python` and `.venv\Scripts\python.exe -m uvicorn` in
place of `.venv/bin/uvicorn` to launch the server.

## Reproduce the experiments

Obtain the FS-Jump3D RGB videos separately and preserve the committed data split.
The [reproducibility guide](docs/REPRODUCIBILITY.md) covers data preparation,
training, validation, test evaluation and artifact verification. The pinned
environment is recorded in [`environments/requirements-current.txt`](environments/requirements-current.txt).

To evaluate the existing augmentation checkpoint reported above, after preparing
the videos and running the model installer:

```bash
.venv/bin/python scripts/evaluate.py --checkpoint models/runs/improvement_phase1/augmentation/best_model.pt --training-summary data/experiments/improvement_phase1/augmentation_summary.json --expected-epoch 30 --output-dir data/reproduction/augmentation_test
```

Use a new output directory. Running `evaluate.py` without arguments selects the
historical baseline, not the current demo model. The baseline checkpoint is not
bundled; train a new baseline to evaluate that configuration.

To train new runs:

```bash
.venv/bin/python scripts/train.py --config configs/train.yaml --run-dir models/runs/reproduction/baseline
.venv/bin/python scripts/train.py --config configs/improvement_phase1_augmentation.yaml --run-dir models/runs/reproduction/augmentation
```

Each run saves its configuration, history, selected checkpoint, source snapshot,
environment metadata and input/result fingerprints. To evaluate a newly trained
run, supply its own checkpoint, `summary.json`, and recorded `best_epoch`.
Retraining can produce different numerical results across seeds and hardware.

## Repository structure

| Path | Contents |
| --- | --- |
| `src/fs_jump3d/` | Data processing, model, training, evaluation, and inference modules |
| `scripts/` | Command-line entry points |
| `configs/` | Experiment settings and protocol descriptions |
| `artifacts/augmentation-v1/` | Evaluated demo checkpoint, checksum and provenance |
| `data/` | Tracked audit, split, and experiment results; local raw data and caches |
| `tests/` | Automated checks |
| `web/` | Web demo interface |
| `docs/` | Reproduction instructions |
| `environments/` | Dependency snapshot |

## Further work

Three directions for further evaluation are:

- **Training stability:** repeat experiments across multiple training seeds and report the mean and standard deviation of validation metrics.
- **Class confusion:** analyze Salchow–Toe Loop and Axel–Lutz errors on validation data and evaluate targeted improvements.
- **Unseen-skater generalization:** use leave-one-skater-out evaluation, reported separately from the current attempt-held-out results.

Use validation data for model development and reserve held-out evaluation for a finalized protocol. The current test set has already been inspected; further test-informed changes require a new independent evaluation design. Confidence intervals should account for camera views belonging to the same physical attempt.

## Data and references

Dataset access, checkpoint availability, and installation are documented in
[Data and model artifacts](docs/DATA_AND_MODELS.md). See [CITATION.cff](CITATION.cff)
for repository citation metadata and cite the original FS-Jump3D paper when using
the dataset. Reproduction instructions and verified environment checks are in the
[reproducibility guide](docs/REPRODUCIBILITY.md) and [validation record](docs/VALIDATION.md).

## License

Project source code and accompanying documentation are licensed under [MIT](LICENSE).
The FS-Jump3D dataset and bundled model weights are excluded from this license;
see [data and model information](docs/DATA_AND_MODELS.md) and the
[checkpoint record](artifacts/augmentation-v1/README.md).
