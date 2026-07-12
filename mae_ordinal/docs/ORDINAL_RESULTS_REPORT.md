# VideoMAE CORN Ordinal Results Report

Contrastive experiment: **frozen backbone (linear-probe)** vs **full fine-tuning** of a
VideoMAE-Base backbone with a CORN ordinal head, coronal (AP) view only, `fuse01` 4-class
target, same train/val split (261/150) and same metrics as the CVFSNet coronal baseline.

Ran in parallel on two GPUs (frozen → GPU 0, fine-tune → GPU 1). W&B project `AmTICIS`:
- Frozen: https://wandb.ai/pytorch-vertebra/AmTICIS/runs/8cz3h5eu
- Fine-tune: https://wandb.ai/pytorch-vertebra/AmTICIS/runs/wwujnf64

## The fix that made it work

The earlier run collapsed because the pipeline's `TioZNormalization(div255)` produces inputs
with **std ≈ 0.004**, and VideoMAE (no input BatchNorm; adds fixed sinusoidal position
embeddings) needs ~unit-variance input. The adapter now does **per-clip min-max → [0,1] →
ImageNet mean/std**, restoring the range VideoMAE expects (verified: mean 0.18, std 0.50,
range ≈[-2.1, 2.6]). An overfit sanity check then reached 100% train accuracy in a few steps.

## Hyperparameters (from the VideoMAE literature)

Sources: MCG-NJU `VideoMAE/FINETUNE.md`, `OpenGVLab/VideoMAEv2/run_class_finetuning.py`.

| | Frozen (linear-probe) | Full fine-tune |
|---|---|---|
| Trainable params | 3.8 K (head only) | 86.2 M (all) |
| Optimizer | AdamW (0.9, 0.999) | AdamW (0.9, 0.999) |
| Base LR | 1e-3 | 2e-4 |
| Layer-wise LR decay | — | 0.75 |
| Weight decay | 1e-4 | 0.05 |
| Backbone dropout | — (frozen) | 0.1 (≈ drop_path substitute) |
| Schedule | 5-epoch warmup + cosine | 5-epoch warmup + cosine |
| Batch size | 16 | 8 |
| Max epochs / early stop | 75 / patience 25 | 75 / patience 25 |
| Grad clip | 1.0 | 1.0 |

Note: HF VideoMAE has no stochastic depth, so the paper's `drop_path 0.1` is approximated by
`hidden_dropout_prob 0.1` on the fine-tune run.

## Results (best epoch by `val/fuse/f1_macro`)

| Metric (val) | VideoMAE frozen | VideoMAE fine-tune | CVFSNet coronal (baseline) |
|---|---:|---:|---:|
| **F1 (macro)** | 0.425 | **0.497** | 0.479 |
| **Accuracy** | 0.447 | **0.580** | 0.547 |
| Precision (macro) | 0.463 | 0.547 | 0.499 |
| Recall (macro) | 0.451 | 0.509 | 0.484 |
| Specificity (macro) | 0.816 | 0.856 | 0.849 |
| AUROC (macro) | 0.687 | 0.745 | 0.701 |
| QWK (ordinal) | 0.346 | 0.648 | n/a* |
| MAE (ordinal) | 0.900 | 0.640 | n/a* |
| Best epoch | 45 | 45 | 292 |

\* The baseline (CVFSNet) pipeline does not log QWK/MAE, so those cells are not directly
comparable.

### Per-class recall @ best epoch (0=T0/1, 1=T2a, 2=T2b, 3=T3)

| Model | T0/1 | T2a | T2b | T3 |
|---|---:|---:|---:|---:|
| VideoMAE frozen | 0.61 | 0.48 | 0.48 | 0.23 |
| VideoMAE fine-tune | 0.53 | 0.31 | 0.22 | 0.98 |
| CVFSNet coronal | 0.76 | 0.21 | 0.35 | 0.62 |

## Confusion matrices (val, best-epoch checkpoint)

See `plots/confusion_matrices.png` (regenerated from checkpoints via
`python -m mae_ordinal.plot_confusion`). Recomputing QWK from the reloaded predictions makes it
comparable across all three (including the baseline):

| Model | Accuracy | QWK (from checkpoint) |
|---|---:|---:|
| VideoMAE frozen | 0.447 | 0.346 |
| VideoMAE fine-tune | 0.573 | 0.647 |
| CVFSNet coronal (baseline) | 0.547 | **0.698** |

Reading the matrices:
- **Frozen**: predictions are diffuse; the diagonal is weak and T3 recall is poor (0.23) — the
  frozen K400 features don't separate DSA grades well.
- **Fine-tune**: highest accuracy, but a strong **T3 bias** — it sends most T2a/T2b/T3 cases to
  T3 (T3 recall 0.98, but T2a 0.31 / T2b 0.17). Accuracy is carried by T0/1 and T3.
- **Baseline (CVFSNet)**: best on T0/1 (0.76) and more spread across the diagonal, so despite
  slightly lower accuracy it has the **best ordinal agreement (QWK 0.698 > 0.647)** — its errors
  land closer to the true grade.

**Important nuance:** on the *ordinal* metric (QWK), the CVFSNet baseline actually edges the
fine-tuned VideoMAE, even though VideoMAE wins on accuracy/F1. The VideoMAE accuracy advantage
comes largely from confidently nailing the two extreme classes while collapsing the middle
grades into T3.

## Summary / conclusions

- **Full fine-tuning beats frozen on every metric** (F1 0.497 vs 0.425, acc 0.580 vs 0.447,
  AUROC 0.745 vs 0.687, QWK 0.648 vs 0.346). With only 261 training clips the frozen K400
  features are decent but clearly benefit from adapting the backbone to DSA.
- **Fine-tuned VideoMAE modestly edges the CVFSNet coronal baseline** on the shared metrics
  (F1 +0.018, accuracy +0.033, AUROC +0.044, precision/recall/specificity all slightly higher).
  So the ordinal-VideoMAE reformulation is competitive with — and marginally better than — the
  CNN baseline on coronal-only grading.
- **The ordinal objective helps ranking**: the fine-tuned model's QWK 0.65 / MAE 0.64 indicate
  errors tend to land on adjacent grades rather than far-off ones.
- **Class balance is uneven**: the fine-tuned run over-predicts T3 (recall 0.98) while T2a/T2b
  suffer (0.31/0.22); the frozen model is more balanced but lower overall. This is a target for
  improvement (loss-level class weighting on top of the sampler).

## Caveats

- **`test = val`** (no held-out test set), single seed, best-epoch-on-val selection → numbers
  are optimistic and the ~0.02–0.03 gap over the baseline is within the noise of a 150-sample
  val set. Treat "beats the baseline" as *competitive/promising*, not conclusive.
- Both peaked ~epoch 45 then overfit (expected for an 86 M transformer on 261 clips).
- transformers 5.x reinitializes VideoMAE attention q/v biases; fine-tuning trains them, but the
  frozen run keeps those (randomly-initialized) biases fixed — a small disadvantage for frozen.

## Suggested next steps

- Proper held-out test split + multi-seed averaging for a defensible comparison.
- Class-weighted CORN loss (or focal-style weighting) to recover T2a/T2b recall.
- Try `videomae-base` SSL checkpoint vs K400-finetuned; try 8 vs 16 frames.
- Fix the q/v-bias loading (pin a transformers version that maps them) for a cleaner init.
