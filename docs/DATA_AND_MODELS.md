# Data and model artifacts

## Dataset

Use the videos download links in the [official FS-Jump3D repository](https://github.com/ryota-skating/FS-Jump3D#download).
Extract the videos so paths follow `data/raw/fs-jump3d/skater_A/cam_1/Axel_1.mp4`.
Keep the committed `data/splits/dataset_split.csv` for comparisons with the reported results.
The six-class experiment uses RGB video, not the C3D or pose JSON downloads.

FS-Jump3D is distributed by its authors under
[CC BY-NC-SA 4.0](https://github.com/ryota-skating/FS-Jump3D#license).
Obtain the videos from the original source and follow its terms;
this repository does not redistribute them.

Dataset reference:

> Ryota Tanaka, Tomohiro Suzuki, and Keisuke Fujii. 2024.
> 3D Pose-Based Temporal Action Segmentation for Figure Skating: A Fine-Grained and Jump Procedure-Aware Annotation Approach.
> MMSports '24, pp. 17–26. https://doi.org/10.1145/3689061.3689077

## Evaluated demo checkpoint

- Architecture: CNN-BiLSTM, six classes.
- Training configuration: `configs/improvement_phase1_augmentation.yaml`.
- Selected epoch: 30, minimum validation loss.
- SHA-256: `8ab4aaa77dc4df660148bcedb35ebae3a520515571e01d5e93edc82f3290b8ce`.
- Report: `data/evaluation/phase1_augmentation_final/final_comparison.md`.
- Installation target: `models/runs/improvement_phase1/augmentation/best_model.pt`.

The checkpoint is not included in Git and no public checkpoint download is
published yet. The application therefore cannot be started from a clone alone.
A local release bundle has been prepared; it is not a hosted release or a grant
of a separate weights license. Publication and weight-specific terms must be
settled before distribution. Reproduction instructions do not replace the terms
applicable to the data and weights.

When you have a copy of the evaluated checkpoint:

```bash
.venv/bin/python scripts/install_model.py /path/to/best_model.pt
.venv/bin/uvicorn --app-dir src fs_jump3d.web:app --host 127.0.0.1 --port 8000
```

The installer verifies the pinned hash before installing and refuses to replace
a different checkpoint. A newly trained model will usually have a different
hash; do not rename it and present it as the evaluated artifact. Preserve its
own configuration, metrics and provenance, and explicitly configure inference
for that model if deploying a new experiment.

## Historical reports and deployment

The evaluation reports record the deployment state at the time of evaluation.
On 2026-11-04 the demo was switched from the epoch-19 baseline to the evaluated
epoch-30 augmentation checkpoint. This was a deployment update, not retraining;
the original reports and predictions remain intact.
