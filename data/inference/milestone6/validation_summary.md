# Milestone 6 Single-Video Inference Validation

The reusable inference pipeline loaded `models/runs/milestone4/best_model.pt` (epoch 19; SHA-256 `31d6fbba519b628798b225c9276c895fdeb318312a2afe8598ac3b406dacdefe`) on MPS. It uses the existing streaming OpenCV preprocessing: 32 uniformly sampled RGB frames resized to 224x224, float32 in [0, 1], then a `[1, 32, 3, 224, 224]` model input. The result contains a predicted label, confidence, six class probabilities, model ID, device, basic video metadata, and timing fields.

These three existing train/validation clips were used only to validate the engineering path. They are not a new model evaluation or a tuning sample.

| Clip | Established split | Prediction | Confidence | Probability sum | Preprocessing | Model | Total per video |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `skater_A/cam_1/Axel_2.mp4` | train | Lutz | 0.634 | 0.999999996 | 0.913s | 0.511s | 1.424s |
| `skater_A/cam_1/Salchow_8.mp4` | val | Salchow | 0.566 | 1.000000051 | 0.945s | 0.012s | 0.958s |
| `skater_A/cam_1/Toeloop_1.mp4` | train | Salchow | 0.695 | 1.000000015 | 0.942s | 0.012s | 0.955s |

The first clip was run a second time through the same loaded predictor. The maximum absolute difference across its six probabilities was `0.0`; the repeat took `0.954s` to preprocess and `0.013s` for the model (`0.967s` total). The checkpoint load itself took about `0.051s` after Python and PyTorch imports. A fresh CLI process may have additional import and first MPS-kernel warmup cost; the separate CLI check took `0.998s` to preprocess and `1.852s` for the first model pass.

The current path is practical for a local single-clip web MVP, with roughly one second per warmed request dominated by video decoding. Milestone 7 should load the predictor once at application startup and reuse it for requests. The web flow requires only one clip, not 12 camera views. The 72.22% group-level test result is a separate evaluation analysis, not this inference mode.

The model classifies already-trimmed single-jump MP4 clips. It does not detect or trim jumps in full programs, classify combinations, estimate rotation count or quality/GOE, or establish performance on arbitrary broadcast, competition, phone, or internet footage. Confidence is a softmax score, not a calibrated reliability guarantee.
