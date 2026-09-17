# Post-MVP Model Improvement: Phase 1

## Scope and baseline diagnosis

All decisions below use the existing grouped **train and validation** splits only. No Phase 1 candidate was run on the test split. The baseline checkpoint and production inference/web configuration remain unchanged.

The scratch-trained 395,222-parameter CNN-BiLSTM uses Adam at a fixed 1e-4 LR, dropout 0.3, no augmentation, no weight decay, and validation-loss early stopping (patience 5). The training split is class-balanced (336 views / 28 physical attempts per class), so class weighting is not supported by class counts. Baseline training accuracy rose to 69.0% by epoch 24 while validation accuracy was 56.0%; best epoch 19 was 67.13%, but epoch 20 fell to 20.60% and epoch 21 recovered to 67.13%. This is strong evidence of validation instability and a later train/validation gap, but does not isolate the cause. The model is neither obviously too large nor obviously too small from parameter count alone. No augmentation and fixed LR are plausible contributors. Prior held-out camera variation indicates viewpoint sensitivity, but this phase does not use held-out results for selection.

The existing epoch-19 checkpoint was scored once on the unchanged validation split to establish macro F1 and per-class reference. Its validation accuracy is 67.13%, macro F1 0.673, and loss 0.7804. Baseline runtime was 2,472 seconds including the first 1,294-second cache build, so direct runtime comparisons should use caution.

## Controlled experiment set

Every candidate kept seed 42, the same train/validation rows, six classes, 32 uniformly sampled 224x224 RGB frames, model architecture, batch size 4, 30-epoch cap, and validation-loss early stopping with patience 5. Validation was always deterministic and unaugmented. Checkpoints are separate and ignored under `models/runs/improvement_phase1/`. Configs live in `configs/improvement_phase1_*.yaml`; histories and metrics are tracked here.

| Experiment | Exact main change | Selected epoch | Selected val accuracy | Peak val accuracy | Selected macro F1 | Best val loss | Runtime | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | Adam 1e-4; no augmentation or schedule | 19/24 | 67.13% | 67.13% | 0.673 | 0.7804 | 2,472s incl. cache build | Strong epoch-to-epoch volatility |
| Augmentation | Clip-consistent translation +/-8 px; gain 0.9-1.1; bias +/-0.04; no flip | 30/30 | 70.37% | 70.37% | 0.701 | 0.7616 | 1,840s | Best aggregate accuracy/F1; Toeloop worsened |
| AdamW | Replace Adam with AdamW, weight decay 1e-4 | 19/24 | 67.13% | 67.13% | 0.673 | 0.7804 | 1,178s | Numerically same logged trajectory as baseline; no useful effect |
| Plateau | Adam + ReduceLROnPlateau on val loss, factor 0.5, patience 2, floor 1e-6 | 29/30 | 68.75% | 70.37% | 0.687 | 0.7341 | 1,286s | Lower selected loss, but lower selected accuracy/F1 than augmentation |
| Augmentation + plateau | Combine the two settings above | 30/30 | 70.37% | 70.37% | 0.701 | 0.7616 | 1,494s | Scheduler never fired; checkpoint SHA matches augmentation exactly |

Selected metrics refer to the **minimum-validation-loss checkpoint**, not the noisiest highest-accuracy epoch. The plateau run peaked at 70.37% accuracy in epoch 26, but its minimum loss occurred in epoch 29 at 68.75% accuracy. The augmentation and combined runs were still improving at the 30-epoch cap; no claim is made that they converged or are stable. Their last-five-epoch validation accuracies ranged from 53.47% to 70.37%.

At the selected epoch, baseline train/validation accuracy was 64.83%/67.13%, augmentation 65.13%/70.37%, and plateau 69.94%/68.75%. Augmented training accuracy is measured on perturbed inputs, so a validation score above train accuracy is not by itself evidence against overfitting. At the baseline's final epoch the gap reversed sharply (69.00% train versus 56.02% validation).

## Validation slices and selection

| Class F1 | Baseline | Augmentation | Plateau |
| --- | ---: | ---: | ---: |
| Axel | 0.921 | 0.942 | 0.897 |
| Flip | 0.600 | 0.679 | 0.692 |
| Loop | 0.753 | 0.786 | 0.727 |
| Lutz | 0.649 | 0.625 | 0.676 |
| Salchow | 0.577 | 0.683 | 0.636 |
| Toeloop | 0.537 | 0.492 | 0.492 |

For 72 validation Toeloop views, Toeloop -> Salchow increased from 19 (baseline) to 29 (augmentation) and 24 (plateau). Conversely, Salchow -> Toeloop fell from 31 to 17 and 23. Aggregate improvement therefore shifts part of the Salchow/Toeloop error rather than solving it. Validation camera accuracy ranged from 55.6%-77.8% for baseline, 41.7%-77.8% for augmentation, and 55.6%-88.9% for plateau; each camera slice has only 36 videos, so this is descriptive and does not establish improved viewpoint robustness.

**Provisional validation winner: augmentation alone**, checkpoint `models/runs/improvement_phase1/augmentation/best_model.pt` (SHA-256 `8ab4aaa77dc4df660148bcedb35ebae3a520515571e01d5e93edc82f3290b8ce`). Versus baseline at selected checkpoints, accuracy improves 3.24 percentage points, macro F1 by 0.028, and loss falls by 0.0189. It has the best aggregate validation accuracy/F1; the combined run is the same model, and plateau trades aggregate scores for lower loss. This is **provisional**, not a convincing broad improvement: Toeloop worsens, camera 11 weakens markedly, validation remains volatile, and the best result is at the epoch cap. The production checkpoint is unchanged. Before considering any held-out test or deployment decision, a repeat seed or follow-up validation-only experiment focused on Toeloop/Salchow and stability is advisable. No test-set evaluation occurred in Phase 1.
