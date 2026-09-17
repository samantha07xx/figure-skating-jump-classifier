# Phase 1 Augmentation Final Held-Out Evaluation

- Checkpoint: `models/runs/improvement_phase1/augmentation/best_model.pt`
- Checkpoint SHA-256: `8ab4aaa77dc4df660148bcedb35ebae3a520515571e01d5e93edc82f3290b8ce`
- Device: `mps`
- Scope: grouped held-out FS-Jump3D test split; no retraining or test-driven model selection

## Video-Level Performance

432 individual camera-view videos; accuracy **71.99%**, macro F1 **0.719**.

| Average | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Macro | 0.737 | 0.720 | 0.719 |
| Weighted | 0.737 | 0.720 | 0.719 |

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Axel | 0.952 | 0.819 | 0.881 | 72 |
| Flip | 0.663 | 0.847 | 0.744 | 72 |
| Loop | 0.830 | 0.542 | 0.655 | 72 |
| Lutz | 0.753 | 0.764 | 0.759 | 72 |
| Salchow | 0.636 | 0.778 | 0.700 | 72 |
| Toeloop | 0.586 | 0.569 | 0.577 | 72 |

## Physical-Attempt Performance

36 attempts, each with 12 views. Aggregation: arithmetic mean of the 12 camera-view class-probability vectors, followed by argmax.
Group accuracy **83.33%**; macro precision **0.855**, recall **0.833**, F1 **0.834**.

## Camera Performance

| Camera | Videos | Accuracy | Macro F1 |
| ---: | ---: | ---: | ---: |
| 1 | 36 | 63.89% | 0.603 |
| 2 | 36 | 77.78% | 0.777 |
| 3 | 36 | 83.33% | 0.826 |
| 4 | 36 | 75.00% | 0.721 |
| 5 | 36 | 77.78% | 0.762 |
| 6 | 36 | 80.56% | 0.803 |
| 7 | 36 | 75.00% | 0.751 |
| 8 | 36 | 66.67% | 0.667 |
| 9 | 36 | 72.22% | 0.720 |
| 10 | 36 | 75.00% | 0.737 |
| 11 | 36 | 50.00% | 0.489 |
| 12 | 36 | 66.67% | 0.668 |

## Skater Performance

| Skater | Videos | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| A | 108 | 65.74% | 0.625 |
| B | 108 | 61.11% | 0.591 |
| C | 108 | 78.70% | 0.806 |
| D | 108 | 82.41% | 0.813 |

Each skater has all six classes, but their class proportions differ; these descriptive scores are not controlled skater comparisons.

## Errors And Confidence

- Incorrect videos: 121
- Strongest class by F1: Axel
- Weakest class by F1: Toeloop
- Mean confidence, correct: 0.679
- Mean confidence, incorrect: 0.576
- High-confidence errors (confidence >= 0.8): 10

| True | Predicted | Count |
| --- | --- | ---: |
| Toeloop | Salchow | 27 |
| Loop | Flip | 17 |
| Salchow | Toeloop | 16 |
| Lutz | Flip | 13 |
| Axel | Lutz | 12 |
| Loop | Toeloop | 11 |
| Flip | Lutz | 6 |
| Loop | Salchow | 5 |
| Toeloop | Loop | 3 |
| Flip | Loop | 3 |

### High-Confidence Mistakes

| Video | True | Predicted | Confidence |
| --- | --- | --- | ---: |
| skater_D/cam_1/Lutz_10.mp4 | Lutz | Axel | 0.914 |
| skater_B/cam_11/Salchow_3.mp4 | Salchow | Toeloop | 0.904 |
| skater_B/cam_7/Salchow_3.mp4 | Salchow | Toeloop | 0.902 |
| skater_D/cam_7/Salchow_1.mp4 | Salchow | Toeloop | 0.898 |
| skater_D/cam_11/Salchow_1.mp4 | Salchow | Toeloop | 0.879 |

## Limitations

FS-Jump3D is controlled, multi-camera footage from four skaters. All four skaters appear in the established train/validation/test split, so this is not unseen-skater evaluation. These results do not establish generalization to broadcast, competition, phone, or internet videos, or to substantially different recording conditions.

The confusion-matrix CSV/PNG files, per-video predictions, and per-group predictions contain the complete results. Observed confusion and viewpoint differences are descriptive; no biomechanical cause is inferred.
