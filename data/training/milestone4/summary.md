# Milestone 4 Training Summary

The first six-class CNN-BiLSTM was trained on the established grouped split. Only train and validation videos were loaded. The held-out test split remains unevaluated.

## Data And Model

- Train: 2,016 videos from 168 physical-attempt groups
- Validation: 432 videos from 36 physical-attempt groups
- Preprocessing: 32 uniformly sampled RGB frames at 224x224, float32 in [0, 1]
- CNN: four Conv2d/BatchNorm/ReLU/MaxPool blocks with channels 3 -> 16 -> 32 -> 64 -> 128, preceded by 4x average pooling, then global average pooling and a 128-dimensional projection
- Temporal model: one-layer bidirectional LSTM, input size 128, hidden size 128 per direction
- Classifier: dropout 0.3, Linear(256, 64), ReLU, dropout 0.3, Linear(64, 6); output is raw logits
- Trainable parameters: 395,222

## Configuration And Outcome

- Seed: 42; device: MPS; batch size: 4; DataLoader workers: 2
- Optimizer: Adam; learning rate: 0.0001; loss: CrossEntropyLoss
- Maximum epochs: 30; early stopping patience: 5 based on validation loss
- Trained epochs: 24; early stopping triggered
- Best epoch: 19; validation loss: 0.7804; validation accuracy: 67.13%
- Final epoch 24: train loss 0.7300, train accuracy 69.00%; validation loss 0.9969, validation accuracy 56.02%
- Total training runtime: 2,472 seconds (41.2 minutes), including a 1,294-second first epoch that decoded and cached all train/validation clips
- Cached epochs 2-24 averaged 51.2 seconds each
- Smoke run: one epoch on four train and four validation clips; forward, backward, two-worker loading, MPS, and checkpoint writing succeeded. Smoke metrics are not model results.

## Artifacts And Interpretation

- Best checkpoint: `models/runs/milestone4/best_model.pt` (ignored by Git)
- Full resolved configuration: `models/runs/milestone4/resolved_config.yaml` (ignored by Git)
- Tracked lightweight records: `data/training/milestone4/history.csv`, `summary.json`, and this summary
- An ignored uint8 frame cache under `data/processed/frame_cache/` avoids repeated full-resolution video decoding. The training path streams frames and retains only the 32 selected frames; a real clip and synthetic fixtures matched the original preprocessing exactly.
- Hyperparameters, split membership, and input shape were unchanged from the initial configuration. No augmentation or class weighting was applied.
- Validation loss and accuracy varied substantially between epochs. Epoch 20, for example, had 20.60% validation accuracy, whereas epoch 19 had 67.13%. The final epoch was worse than the best checkpoint, and the train/validation gap suggests overfitting or unstable generalization.
- Results describe validation performance on a controlled four-skater, multi-camera dataset. They do not imply performance on unseen skaters or arbitrary skating footage.
