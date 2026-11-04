# Evaluated augmentation checkpoint

This is the exact checkpoint used for the reported augmentation-model evaluation,
provided for reproducing that experiment and running the research demo.

- Model: CNN-BiLSTM, 395,222 trainable parameters.
- Configuration: `configs/improvement_phase1_augmentation.yaml`.
- Selection: epoch 30, minimum validation loss.
- Test report: `data/evaluation/phase1_augmentation_final/metrics.json`.
- SHA-256: `8ab4aaa77dc4df660148bcedb35ebae3a520515571e01d5e93edc82f3290b8ce`.
- Original checkpoint format is preserved, including optimizer state and model/preprocessing metadata.
- No raw videos are bundled. The baseline checkpoint is not included.

Install from the repository root with `python scripts/install_model.py`.
Only load checkpoint files from a trusted source; the installer checks the pinned
hash before the application loads the model.

Training data: [FS-Jump3D](https://github.com/ryota-skating/FS-Jump3D), by Ryota Tanaka,
Tomohiro Suzuki and Keisuke Fujii. Dataset paper:
[3D Pose-Based Temporal Action Segmentation for Figure Skating: A Fine-Grained and Jump Procedure-Aware Annotation Approach](https://doi.org/10.1145/3689061.3689077), MMSports 2024.
The authors distribute the dataset under CC BY-NC-SA 4.0. Obtain data from their
official repository and follow its terms. This checkpoint is a separately trained
classification model, not an official FS-Jump3D model or an implementation of the
paper's temporal segmentation method. No MIT license is asserted for this artifact.
