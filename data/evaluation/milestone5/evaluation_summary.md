# Milestone 5 Held-Out Evaluation

- Checkpoint: `models/runs/milestone4/best_model.pt`
- Checkpoint SHA-256: `31d6fbba519b628798b225c9276c895fdeb318312a2afe8598ac3b406dacdefe`
- Device: `mps`
- Scope: grouped held-out FS-Jump3D test split; no retraining or test-driven model selection

## Video-Level Performance

432 individual camera-view videos; accuracy **65.97%**, macro F1 **0.652**.

| Average | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Macro | 0.660 | 0.660 | 0.652 |
| Weighted | 0.660 | 0.660 | 0.652 |

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Axel | 0.894 | 0.819 | 0.855 | 72 |
| Flip | 0.656 | 0.556 | 0.602 | 72 |
| Loop | 0.710 | 0.681 | 0.695 | 72 |
| Lutz | 0.648 | 0.819 | 0.724 | 72 |
| Salchow | 0.561 | 0.764 | 0.647 | 72 |
| Toeloop | 0.489 | 0.319 | 0.387 | 72 |

## Physical-Attempt Performance

36 attempts, each with 12 views. Aggregation: arithmetic mean of the 12 camera-view class-probability vectors, followed by argmax.
Group accuracy **72.22%**; macro precision **0.715**, recall **0.722**, F1 **0.704**.

## Camera Performance

| Camera | Videos | Accuracy | Macro F1 |
| ---: | ---: | ---: | ---: |
| 1 | 36 | 69.44% | 0.684 |
| 2 | 36 | 80.56% | 0.800 |
| 3 | 36 | 66.67% | 0.659 |
| 4 | 36 | 69.44% | 0.679 |
| 5 | 36 | 72.22% | 0.715 |
| 6 | 36 | 69.44% | 0.680 |
| 7 | 36 | 47.22% | 0.458 |
| 8 | 36 | 52.78% | 0.524 |
| 9 | 36 | 66.67% | 0.668 |
| 10 | 36 | 72.22% | 0.711 |
| 11 | 36 | 61.11% | 0.597 |
| 12 | 36 | 63.89% | 0.619 |

## Skater Performance

| Skater | Videos | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| A | 108 | 68.52% | 0.621 |
| B | 108 | 66.67% | 0.652 |
| C | 108 | 59.26% | 0.610 |
| D | 108 | 69.44% | 0.693 |

Each skater has all six classes, but their class proportions differ; these descriptive scores are not controlled skater comparisons.

## Errors And Confidence

- Incorrect videos: 147
- Strongest class by F1: Axel
- Weakest class by F1: Toeloop
- Mean confidence, correct: 0.666
- Mean confidence, incorrect: 0.609
- High-confidence errors (confidence >= 0.8): 19

| True | Predicted | Count |
| --- | --- | ---: |
| Toeloop | Salchow | 43 |
| Flip | Lutz | 18 |
| Salchow | Toeloop | 16 |
| Loop | Flip | 14 |
| Flip | Loop | 14 |
| Axel | Lutz | 13 |
| Loop | Toeloop | 8 |
| Lutz | Axel | 7 |
| Lutz | Flip | 6 |
| Toeloop | Loop | 5 |

### High-Confidence Mistakes

| Video | True | Predicted | Confidence |
| --- | --- | --- | ---: |
| skater_B/cam_8/Salchow_3.mp4 | Salchow | Toeloop | 0.912 |
| skater_B/cam_7/Salchow_3.mp4 | Salchow | Toeloop | 0.900 |
| skater_D/cam_1/Lutz_10.mp4 | Lutz | Axel | 0.896 |
| skater_D/cam_12/Lutz_10.mp4 | Lutz | Axel | 0.862 |
| skater_D/cam_7/Toeloop_3.mp4 | Toeloop | Salchow | 0.857 |

## Limitations

FS-Jump3D is controlled, multi-camera footage from four skaters. All four skaters appear in the established train/validation/test split, so this is not unseen-skater evaluation. These results do not establish generalization to broadcast, competition, phone, or internet videos, or to substantially different recording conditions.

The confusion-matrix CSV/PNG files, per-video predictions, and per-group predictions contain the complete results. Observed confusion and viewpoint differences are descriptive; no biomechanical cause is inferred.
