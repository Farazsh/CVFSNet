# MedGemma QLoRA Binary TICI — Tuning Results (E2, AP view)

Results from the four-seed QLoRA training wave described in
[QLORA_AP_EXPERIMENT.md](QLORA_AP_EXPERIMENT.md). Every number here comes from
the **53-study tuning split**. The 150-study locked final split has not been
read; `--phase final` has not been run.

Class 0 is `T012a` (mTICI 0/1/2A), class 1 is `T2b3` (mTICI 2B/3). The tuning
split holds 26 class-0 and 27 class-1 studies.

## 1. Headline

QLoRA moves MedGemma from no discrimination to real discrimination. Zero-shot
MedGemma scored AUROC 0.403 on this same split; after fine-tuning, every seed
lands between 0.742 and 0.868.

But the fine-tuned model learns a usable **ranking** without learning a usable
**probability scale**, and the scale it does learn differs from seed to seed.
That distinction drives everything below, and it determines how the final split
should be read.

## 2. Per-seed results at the selected epoch

Epoch selection is macro F1 at the tuned threshold, then AUROC as tie-breaker,
with early-stopping patience 6. AUROC confidence intervals are 10,000
stratified bootstrap replicates over the 53 tuning studies.

| Seed | Best epoch | AUROC | AUROC 95% CI | Macro F1 @ tuned | Macro F1 @ 0.5 | Balanced acc. | Tuned threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 10 | 0.868 | 0.761–0.950 | 0.828 | 0.826 | 0.828 | 0.754 |
| 14207 | 11 | 0.791 | 0.658–0.905 | 0.792 | 0.523 | 0.572 | 0.086 |
| 1337 | 10 | 0.788 | 0.657–0.897 | 0.767 | 0.679 | 0.679 | 0.191 |
| 2718 | 10 | 0.742 | 0.594–0.872 | 0.752 | 0.329 | 0.500 | 0.148 |
| **mean ± sd** | | **0.797 ± 0.052** | | **0.785 ± 0.033** | **0.589 ± 0.213** | **0.645 ± 0.142** | **0.295 ± 0.309** |

Read the last three columns together. Macro F1 at the tuned threshold is stable
across seeds (0.752–0.828, sd 0.033). Macro F1 at the fixed 0.5 threshold is
not: it ranges from 0.329 to 0.826, sd 0.213 — six times the spread. The
difference between those two columns is entirely the threshold.

## 3. Why the 0.5 threshold collapses

The four seeds put their probabilities in completely different places:

| Seed | Probability range | Mean probability | Fraction above 0.5 | AUROC |
|---|---|---:|---:|---:|
| 42 | 0.000 – 1.000 | 0.626 | 0.64 | 0.868 |
| 1337 | 0.001 – 0.995 | 0.516 | 0.49 | 0.788 |
| 14207 | 0.002 – 0.909 | 0.228 | 0.19 | 0.791 |
| 2718 | **0.091 – 0.307** | 0.174 | **0.00** | 0.742 |

Seed 2718 never emits a probability above 0.307. At the fixed 0.5 threshold it
predicts class 0 for all 53 studies, which is why its balanced accuracy is
exactly 0.500 and its macro F1 is 0.329 — a degenerate classifier. Yet its
AUROC is 0.742: it *ranks* the successful-reperfusion studies above the
unsuccessful ones perfectly respectably. It has simply squeezed the entire
ranking into a 0.09–0.31 band. Seed 14207 is a milder version of the same
thing.

So the sigmoid output is not a probability in any meaningful sense; it is a
monotone score whose scale is an artifact of the seed. Calibration metrics say
the same thing: mean expected calibration error 0.268 ± 0.087, mean Brier score
0.261 ± 0.078, both far from what a calibrated model on a balanced binary task
would produce.

## 4. The seeds agree on the ranking

Despite the scale disagreement, the seeds largely rank the same studies the same
way. Spearman rank correlation between seeds' tuning probabilities:

| | 42 | 1337 | 2718 |
|---|---:|---:|---:|
| **14207** | 0.806 | 0.833 | 0.632 |
| **42** | | 0.901 | 0.618 |
| **1337** | | | 0.556 |

Seeds 14207, 42, and 1337 are tightly correlated (0.81–0.90). Seed 2718 — the
weakest and the one with the collapsed probability range — is the outlier at
0.56–0.63, so it is learning a partly different ordering, not just a different
scale.

Averaging the four seeds' probabilities gives an ensemble at AUROC 0.805
(95% CI 0.675–0.913), macro F1 0.806 at a tuned threshold of 0.282. That is
within noise of the best single seed and does not justify itself as a separate
model; it is reported here only to show the seeds are not averaging away into
mush.

## 5. The selection metric stops earlier than AUROC would

Tuning AUROC by epoch, seed-averaged (`*` marks each seed's selected epoch):

| Epoch | 14207 | 42 | 1337 | 2718 | mean |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.648 | 0.359 | 0.607 | 0.724 | 0.584 |
| 5 | 0.714 | 0.749 | 0.781 | 0.664 | 0.727 |
| 8 | 0.785 | 0.859 | 0.744 | 0.761 | 0.787 |
| 10 | 0.808 | 0.868\* | 0.788\* | 0.742\* | 0.801 |
| 11 | 0.791\* | 0.846 | 0.768 | 0.761 | 0.791 |
| 13 | 0.819 | 0.863 | 0.815 | 0.768 | 0.816 |
| 16 | 0.842 | 0.858 | 0.818 | 0.752 | 0.817 |
| 17 | 0.855 | — | — | — | — |

Every seed was selected at epoch 10 or 11, but mean AUROC keeps climbing after
that and is still at its highest when early stopping fires (0.817 at epoch 16;
seed 14207 reaches 0.855 at epoch 17). Meanwhile the model is plainly
overfitting in loss:

| Point | Train loss | Tuning loss | Gap | Tuning AUROC |
|---|---:|---:|---:|---:|
| Epoch 1 | 0.754 | 0.769 | 0.016 | 0.584 |
| Selected epoch | 0.503 | 0.862 | 0.359 | 0.797 |
| Last epoch | 0.133 | 1.544 | 1.411 | 0.821 |

The model's *ranking* keeps improving while its *probabilities* diverge — it
grows more confident and more wrong in probability space, which inflates BCE
loss and wrecks calibration, but does not damage the ordering. Because the
selection metric embeds a threshold, it tracks the probability collapse rather
than the ranking, and stops early relative to AUROC. Early stopping fired on
that metric at epoch 16–17 in all four runs, so no run terminated because AUROC
plateaued.

This does not mean the selection rule was wrong. It means the rule optimizes a
thresholded metric and should be understood as doing so. Whether epoch 10 or
epoch 16 is the better checkpoint depends on whether the final split is scored
on ranking (AUROC, where later is better) or on a thresholded decision (macro
F1, where the tuned threshold has to come from somewhere anyway). The committed
`selection.yaml` files already lock in epoch 10/11 and the per-seed thresholds,
and changing that now would mean re-selecting on the tuning split after seeing
these results — which is exactly the bias the protocol is designed to prevent.
The locked choice stands.

## 6. Comparison

| Model | View | Split | AUROC | Macro F1 |
|---|---|---|---:|---:|
| MedGemma zero-shot (best variant) | AP | 53 tuning | 0.403 | 0.440 |
| MedSigLIP zero-shot (best variant) | AP | 53 tuning | 0.516 | 0.396 |
| **MedGemma QLoRA (4 seeds)** | AP | 53 tuning | **0.797 ± 0.052** | **0.785 ± 0.033** |
| VideoMAE full fine-tune (1 seed) | AP | 150 val | 0.925 | 0.826 |

The zero-shot rows are directly comparable: same 53 studies, same split seed,
same preprocessing. Fine-tuning is a large and unambiguous gain over both.

The VideoMAE row is **not** directly comparable and should not be read as
"VideoMAE beats MedGemma by 0.13 AUROC." It is a different split (150 studies,
its own `val`, with `test = val` and best-epoch-on-val selection, single seed),
so its number carries the optimistic bias of selection-on-the-evaluation-set
that this experiment's protocol specifically avoids. The honest comparison is
MedGemma QLoRA on the locked 150 versus VideoMAE on a genuinely held-out set,
and neither exists yet.

## 7. Cost

Per seed, one RTX 3090, four seeds run concurrently on four GPUs:

| Quantity | Value |
|---|---|
| Epochs run before early stop | 16–17 |
| Wall clock | 6.4–6.8 h per seed (~6.8 h for the wave) |
| Training step | 6.9 s/study |
| Peak GPU memory | 6.41 GiB |
| Trainable parameters | 29,810,177 of 1,615,104,881 (1.85%) |

## 8. What this means for the final split

Three points to carry into `--phase final`:

1. **AUROC is the metric to trust.** It is the only reported quantity that is
   invariant to the per-seed probability scale, and it is the only one that is
   stable across seeds (sd 0.052 versus sd 0.213 for macro F1 at 0.5).
2. **Never report macro F1 at 0.5.** For seed 2718 it measures nothing but the
   fact that the sigmoid saturated low. The per-seed tuned threshold in
   `selection.yaml` is the only defensible operating point, and it was chosen
   before the final split was touched, which is what makes it usable.
3. **The tuned macro F1 is optimistically biased even so.** The epoch and the
   threshold were both selected on the same 53 studies, so 0.785 ± 0.033 is a
   fitted number, not an estimate of held-out performance. Expect the locked
   split to come in lower; a drop of a few points would be unremarkable rather
   than a sign that something broke.

## 9. Limitations

- 53 tuning studies is a small selection set. Every confidence interval above is
  roughly 0.25 AUROC wide, and the seed-to-seed AUROC spread (0.742–0.868) sits
  entirely inside a single seed's interval — the seeds are not distinguishable
  from each other on this much data.
- 208 development studies is small for 29.8M trainable parameters, and the
  train/tuning loss gap of 1.4 by the last epoch confirms the model has the
  capacity to memorize them.
- AP view only. The plan's primary configuration is dual-view, which is not run.
- The uncalibrated, seed-dependent probability scale means this model cannot
  currently be used for anything that needs a probability (risk stratification,
  abstention, cost-weighted decisions) — only for ranking. Fixing that would
  need a calibration step (temperature scaling on a split that is not the tuning
  split), which the current protocol has no data left to pay for.
- MedGemma is not clinically validated for reperfusion grading.
