# One-Time Phase 1 Held-Out Evaluation

The Phase 1 validation-selected **augmentation-only** checkpoint was evaluated once on the unchanged FS-Jump3D grouped test split with the Milestone 5 protocol. No retraining, tuning, split changes, or other Phase 1 checkpoints were evaluated on test. This result did not change the production web model.

- Checkpoint: `models/runs/improvement_phase1/augmentation/best_model.pt`
- SHA-256: `8ab4aaa77dc4df660148bcedb35ebae3a520515571e01d5e93edc82f3290b8ce`
- Selection: validation-loss minimum at epoch 30, before test evaluation
- Test set: 432 videos, 36 complete physical-attempt groups, 12 camera views each
- Protocol: deterministic 32-frame 224x224 preprocessing; batch size 4; no test augmentation; per-video softmax; per-attempt arithmetic mean of 12 probability vectors and argmax

## Video-level results

- Accuracy: **71.99%** (311/432)
- Macro precision: **0.737**
- Macro recall: **0.720**
- Macro F1: **0.719**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Axel | 0.952 | 0.819 | 0.881 | 72 |
| Flip | 0.663 | 0.847 | 0.744 | 72 |
| Loop | 0.830 | 0.542 | 0.655 | 72 |
| Lutz | 0.753 | 0.764 | 0.759 | 72 |
| Salchow | 0.636 | 0.778 | 0.700 | 72 |
| Toeloop | 0.586 | 0.569 | 0.577 | 72 |

Confusion matrix, rows true and columns predicted, both in canonical class order:

| True / Predicted | Axel | Flip | Loop | Lutz | Salchow | Toeloop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Axel | 59 | 0 | 1 | 12 | 0 | 0 |
| Flip | 0 | 61 | 3 | 6 | 0 | 2 |
| Loop | 0 | 17 | 39 | 0 | 5 | 11 |
| Lutz | 3 | 13 | 1 | 55 | 0 | 0 |
| Salchow | 0 | 0 | 0 | 0 | 56 | 16 |
| Toeloop | 0 | 1 | 3 | 0 | 27 | 41 |

Toeloop -> Salchow remains the largest error (27 videos), though the original baseline had 43. Salchow -> Toeloop is 16, unchanged from baseline. Toeloop F1 rose from 0.387 to 0.577 but is still the lowest class F1.

## Camera and skater slices

| Camera | Accuracy | Macro F1 | Camera | Accuracy | Macro F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 63.89% | 0.603 | 7 | 75.00% | 0.751 |
| 2 | 77.78% | 0.777 | 8 | 66.67% | 0.667 |
| 3 | 83.33% | 0.826 | 9 | 72.22% | 0.720 |
| 4 | 75.00% | 0.721 | 10 | 75.00% | 0.737 |
| 5 | 77.78% | 0.762 | 11 | 50.00% | 0.489 |
| 6 | 80.56% | 0.803 | 12 | 66.67% | 0.668 |

Each camera has 36 test videos. The 50.00%-83.33% range remains substantial and these small slices are descriptive.

| Skater | Videos | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| A | 108 | 65.74% | 0.625 |
| B | 108 | 61.11% | 0.591 |
| C | 108 | 78.70% | 0.806 |
| D | 108 | 82.41% | 0.813 |

## Physical-attempt and baseline comparison

The 12-view probability-averaged group result is **83.33% accuracy** (30/36) and **0.834 macro F1**. It is a separate multi-view measure, not the single-video web use case.

| Metric | Original Milestone 4 model | Augmentation model | Change |
| --- | ---: | ---: | ---: |
| Video accuracy | 65.97% | 71.99% | **+6.02 percentage points** |
| Video macro F1 | 0.652 | 0.719 | +0.068 |
| Group accuracy | 72.22% | 83.33% | +11.11 percentage points |
| Group macro F1 | 0.704 | 0.834 | +0.130 |

The validation improvement generalized to this held-out grouped test split in aggregate. It does not establish unseen-skater or uncontrolled-video generalization: all four skaters appear across the established splits, camera results vary, and clips are already trimmed. No further model-development decision was made from this test result. The original baseline checkpoint/results and the augmentation checkpoint/results are preserved separately. Production inference still uses `models/runs/milestone4/best_model.pt`.
