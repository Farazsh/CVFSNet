# MedGemma QLoRA on the Val Protocol — AP / Sagittal / Dual

**Status: complete.** Three QLoRA fine-tunes of MedGemma 1.5 4B for binary mTICI
classification (class 0 = mTICI 0/1/2A, class 1 = mTICI 2B/3), one per view configuration,
executed per `QLORA_VAL_PROTOCOL_PLAN.md`. Code at commit `2812b6e`; runs completed
2026-07-13.

---

## 1. Headline result


| Model                       | AUROC                    | Macro F1 @ 0.5 | Balanced acc. | Accuracy |
| --------------------------- | ------------------------ | -------------- | ------------- | -------- |
| MedGemma QLoRA **AP**       | **0.804** [0.730, 0.871] | 0.692          | 0.701         | 0.693    |
| MedGemma QLoRA **sagittal** | **0.845** [0.777, 0.905] | 0.773          | 0.775         | 0.773    |
| MedGemma QLoRA **dual**     | **0.857** [0.794, 0.914] | 0.767          | 0.769         | 0.767    |


**The three views are statistically indistinguishable.** The AP→sagittal gap is 0.041
AUROC and the sagittal→dual gap is 0.012, against bootstrap 95% CIs roughly ±0.07 wide.
Both gaps sit well inside that band. This threshold for calling a difference real was
**fixed in advance** (plan §9.1, "any difference below ~0.08 AUROC between the three views
is not a result"), before any of these numbers existed. It is applied here as written.

The honest conclusion is therefore: *at n=150 with one seed, this experiment cannot
distinguish the three view configurations.* Dual's nominal lead is not evidence that
cross-view reasoning helps.

### Against the other model families

All rows below are the same 150-study val split, scored at the natural threshold. Baseline
values are taken from the project's metrics tables of record —
`Assets/confusion_matrix_metrics_cvfsnet_binary_runs_mod3.csv` and
`..._videomae_binary_runs.csv` — which is where the MedGemma rows now also live.


| Model              | Views        | AUROC     | AUPRC | Accuracy  | Macro F1  | Recall | Precision | Specificity |
| ------------------ | ------------ | --------- | ----- | --------- | --------- | ------ | --------- | ----------- |
| CVFSNet            | Fusion       | **0.928** | 0.905 | **0.907** | **0.907** | 0.929  | 0.878     | 0.887       |
| CVFSNet            | Coronal      | 0.916     | 0.892 | 0.887     | 0.887     | 0.914  | 0.853     | 0.863       |
| CVFSNet            | Sagittal     | 0.890     | 0.859 | 0.833     | 0.833     | 0.829  | 0.817     | 0.838       |
| VideoMAE           | AP           | 0.925     | 0.912 | 0.827     | 0.826     | 0.971  | 0.739     | 0.700       |
| VideoMAE           | Dual         | 0.895     | 0.834 | 0.853     | 0.853     | 0.971  | 0.773     | 0.750       |
| VideoMAE           | Sagittal     | 0.845     | 0.779 | 0.780     | 0.779     | 0.900  | 0.708     | 0.675       |
| **MedGemma QLoRA** | Dual         | 0.857     | 0.840 | 0.767     | 0.767     | 0.800  | 0.727     | 0.738       |
| **MedGemma QLoRA** | Sagittal     | 0.845     | 0.852 | 0.773     | 0.773     | 0.800  | 0.737     | 0.750       |
| **MedGemma QLoRA** | Coronal (AP) | 0.804     | 0.779 | 0.693     | 0.692     | 0.814  | 0.633     | 0.588       |


**MedGemma is the weakest of the three model families on every metric.** The two strongest
baselines — CVFSNet fusion (AUROC 0.928) and VideoMAE AP (0.925) — lie **outside the upper
bound of the 95% CI of *every* MedGemma run**, including MedGemma's best (dual, CI upper
0.914). That gap is real, not noise, and it is the central negative finding of this wave.

The comparison is softer further down: MedGemma sagittal (0.845) matches VideoMAE sagittal
exactly, and CVFSNet sagittal (0.890) sits inside MedGemma dual's CI. So MedGemma is
competitive with the *weakest* configurations of the other families, and clearly beaten by
their best.

The gap is widest on the **thresholded** metrics, not AUROC. MedGemma's best accuracy
(0.773) trails CVFSNet fusion by 13.4 points and VideoMAE dual by 8.6; its AP run trails
CVFSNet coronal by 19.4 points. Its AUROC deficit is smaller than its accuracy deficit,
which is the signature of a model that *ranks* acceptably but whose decision probabilities
are poorly calibrated. See §7 — this is not a small caveat, it is most of the gap.

> **Provenance note.** An earlier draft of this report quoted CVFSNet at AUROC 0.895–0.905
> and accuracy 0.853–0.867, taken from `BINARY_TICI_EXPERIMENT_PLAN.md`. Those values are
> stale. The metrics CSV above, regenerated from the CVFSNet checkpoints, gives
> 0.890–0.928 AUROC and 0.833–0.907 accuracy. The correction moves CVFSNet **up**, and so
> strengthens rather than weakens the conclusion that MedGemma trails it.

> **Bias statement — applies to every row in both tables above.** Each model, MedGemma and
> baselines alike, was early-stopped on the same 150 val studies it reports. There is no
> held-out test set anywhere in this project. Every number here is optimistically biased by
> best-epoch-on-eval selection. That shared bias is deliberate — it is what makes the rows
> comparable — but **none of these are estimates of held-out performance.**



### 1.1 Against MedGemma zero-shot — what the fine-tuning bought


| Run                                       | Split  | n   | AUROC     | AUPRC | Accuracy | Macro F1 | Recall | Specificity |
| ----------------------------------------- | ------ | --- | --------- | ----- | -------- | -------- | ------ | ----------- |
| Zero-shot, `concise_v2` (selected)        | tuning | 53  | **0.403** | 0.451 | 0.453    | 0.440    | 0.296  | 0.615       |
| Zero-shot, `clinical_v1` (prompt-matched) | tuning | 53  | 0.530     | 0.533 | 0.528    | 0.379    | 1.000  | 0.038       |
| QLoRA dual (fine-tuned)                   | val    | 150 | 0.857     | 0.840 | 0.767    | 0.767    | 0.800  | 0.738       |


> **These rows are not on the same studies.** The zero-shot numbers are the 53-study
> *tuning* split, not the 150-study val split. **No valid zero-shot result on val exists:**
> the only zero-shot run that ever consumed val is an invalid run whose every class score is
> `-1e9` (see `ZERO_SHOT_FAILED_RUN_REPORT.md`), and whose numbers must never be cited. The
> comparison below is therefore directional, not a like-for-like delta.

Even allowing for that, the direction is unambiguous: **zero-shot MedGemma has no usable
signal on this task.** The variant the zero-shot tuning rule selected scores AUROC 0.403 —
*below chance*. The variant using the same `clinical_v1` prompt and `percentile_1_99`
scaling as these QLoRA runs scores 0.530, which is chance, and reaches it degenerately by
calling 52 of 53 studies positive (recall 1.000, specificity 0.038).

So the ~0.86 AUROC of the fine-tuned model is not a pretrained MedGemma capability being
surfaced by a good prompt. It is **created by the QLoRA adaptation of 29.8 M parameters**.
That is a real result, and it is the strongest argument for pushing further on adaptation
(§10.2: unfreezing the projector and the vision tower) rather than on prompt engineering.

---



## 2. Protocol


|                            |                                                                |
| -------------------------- | -------------------------------------------------------------- |
| Train                      | 261 studies (the full `train` split)                           |
| Early stopping + reporting | 150 studies (the `val` split) — **the same studies**           |
| Test                       | none                                                           |
| Selection metric           | val AUROC (max), patience 6, max 25 epochs                     |
| Seed                       | 14207 (one seed; split seed also 14207, fixed across all runs) |
| Class balance              | val: 80 negative (T012a) / 70 positive (T2b3); train: 128/133  |


This mirrors the protocol VideoMAE and CVFSNet already use, which is the whole point — it
makes the three model families directly comparable. The cost is that the reported epoch is
chosen on the reported studies.

**Why AUROC and not F1 as the monitor.** Wave 1 showed the sigmoid's *scale* is a seed
artifact: seed 2718 never emitted a probability above 0.307 and was degenerate at a 0.5
threshold (balanced accuracy exactly 0.500) while still scoring AUROC 0.742. Any
thresholded selection metric tracks that probability collapse rather than the model's
discrimination. AUROC is threshold-free and was the only seed-stable quantity in wave 1
(sd 0.052, vs 0.213 for macro F1 @ 0.5). §7 shows this choice was vindicated — AP saturated
its sigmoid exactly as feared, and an F1 monitor would have chased that collapse.

---



## 3. Hyperparameters

Unchanged from wave 1 (`config_qlora_ap.yaml`), which follows Google's official MedGemma
LoRA fine-tuning recipe. All three runs are identical except for `input.views`.


| Group            | Setting                    | Value                                                                      |
| ---------------- | -------------------------- | -------------------------------------------------------------------------- |
| **Quantization** | bits / type                | 4-bit NF4, double quantization                                             |
|                  | compute dtype              | bfloat16                                                                   |
| **LoRA**         | rank / alpha / dropout     | 16 / 16 / 0.05                                                             |
|                  | target scope               | `all_linear_language` — all 238 Linear modules in the language model       |
|                  | vision tower               | **frozen** (asserted at startup)                                           |
|                  | projector                  | frozen                                                                     |
| **Optimizer**    | AdamW, adapter LR          | 2e-4                                                                       |
|                  | head LR                    | 1e-3                                                                       |
|                  | weight decay               | 0.01                                                                       |
| **Schedule**     | cosine, warmup ratio       | 0.03                                                                       |
|                  | grad clip                  | 0.3                                                                        |
| **Batching**     | micro-batch × accumulation | 1 × 8 (effective 8)                                                        |
| **Training**     | max epochs / patience      | 25 / 6                                                                     |
|                  | gradient checkpointing     | on (language model only; tower is frozen so its checkpointing is disabled) |
| **Head**         | dropout                    | 0.2                                                                        |
| **Input**        | frames / size / views      | 8 / 896×896 / per-run                                                      |
|                  | intensity scaling          | percentile 1–99                                                            |
|                  | prompt                     | `clinical_v1`                                                              |
| **Loss**         |                            | `BCEWithLogitsLoss`, no class weighting, no weighted sampler               |


Batch size is structurally 1: the head reads `last_hidden_state[:, -1, :]`, which is only
the last *prompt* token because there is no padding.

### Parameters tuned


|                    | Count          |
| ------------------ | -------------- |
| Trainable          | **29,810,177** |
| Total              | 1,615,104,881  |
| Trainable fraction | **1.85%**      |


Counted directly from the saved checkpoints: **29,802,496** LoRA adapter weights (476
tensors — an A and a B matrix for each of the 238 targeted Linear modules) plus **7,681**
classification-head weights (LayerNorm 2560 → 5,120, Linear 2560→1 → 2,561). These sum to
exactly 29,810,177.

**This count is identical for all three runs.** View count changes only how many images
enter the prompt, not the parameter set — the vision tower and projector are excluded from
adaptation, so AP, sagittal and dual all train exactly the same 29,810,177 weights. A
differing count would have meant something broke; it was checked.

---



## 4. How the DSA scans reach the model

```
NIfTI DSA series (variable H × W × T)
   │
   ├─ view resolution: coronal base name, marker substitution
   │     AP        → *_C*.nii.gz
   │     sagittal  → *_S*.nii.gz          (verified by path AND by pixel content)
   │
   ├─ ResizeView: trilinear resample → (1, 8, 896, 896)
   │     8 frames, chronological; 896 is MedGemma's native SigLIP-400M resolution
   │     (feeding smaller would just be upsampled back by the processor)
   │
   ├─ augmentation (train split only): RandomRotation(±10°, p=0.25),
   │     Crop(p=0.5), Resize — frame-consistent geometry only, one angle per clip,
   │     so no artificial temporal motion. Val/eval split: no augmentation.
   │
   ├─ scan_to_rgb_frames: percentile 1–99 intensity scaling computed ONCE per scan
   │     (shared across all frames, so a bright late contrast phase is not
   │     renormalized away frame-by-frame) → uint8 → 3-channel RGB
   │
   └─ chat template → 256 vision tokens per image
```

The prompt is a single user message:

> **Preamble:** "Review this chronological digital subtraction angiography series. Class 0
> is unsuccessful reperfusion (mTICI 0, 1, or 2A). Class 1 is successful reperfusion (mTICI
> 2B or 3)."
>
> Then, per view: `AP VIEW`, then for each frame `FRAME i OF 8` followed by the image.
> Dual appends `SAGITTAL VIEW` and its 8 frames after the AP block.
>
> **Question:** "Which final reperfusion class is shown? Reply with only 0 or 1."

Frame markers are generated from the actual frame count, never hard-coded in the template.


| Run      | Views            | Images / study | Vision tokens | Approx. prompt length |
| -------- | ---------------- | -------------- | ------------- | --------------------- |
| AP       | `[AP]`           | 8              | 2,048         | ~2,200 tokens         |
| Sagittal | `[sagittal]`     | 8              | 2,048         | ~2,200 tokens         |
| Dual     | `[AP, sagittal]` | **16**         | **4,096**     | ~4,300 tokens         |


**Dual is one datapoint per study, not two.** A single 16-image prompt produces a single
probability. Nothing is averaged and the dataset is not expanded per view — so dual is a
genuine cross-view model, not an ensemble.

## How the output is collected

The generative head is discarded entirely. Instead:

```
trunk(**inputs) → last_hidden_state[:, -1, :]      # final prompt token, 2560-dim
    → LayerNorm(2560)
    → Dropout(0.2)
    → Linear(2560 → 1)                             # float32
    → logit  →  sigmoid  →  P(class 1 = T2b3)
```

The model never generates a token. The class decision is `P(T2b3) ≥ threshold`. Logits are
asserted finite at every step. Training optimizes BCE against the binary label.

---



## 5. Compute and timing

All runs: one RTX 3090 (24 GiB) each, three concurrently on one host. torch 2.6.0+cu124,
transformers 4.57.1, Python 3.12.3, MedGemma revision `91850547`.


| Run      | Epochs run         | Selected epoch | Wall clock | **Time / epoch** | Peak GPU |
| -------- | ------------------ | -------------- | ---------- | ---------------- | -------- |
| AP       | 21 (early-stopped) | 15             | 12.57 h    | **35.9 min**     | 6.41 GiB |
| Sagittal | 24 (early-stopped) | 18             | 14.36 h    | **35.9 min**     | 6.41 GiB |
| Dual     | 21 (early-stopped) | 15             | 27.26 h    | **77.9 min**     | 8.08 GiB |


Wave wall clock, gated by dual: **~27 h** (all three ran in parallel).

**Per-epoch time is remarkably stable** — AP's 21 epochs span 2152.2–2156.7 s, a spread of
4.5 s (0.2%); dual's span 8.9 s. The workload is fixed-size and deterministic, so this is
expected rather than suspicious.

**Per-step time.** One epoch = 261 training studies + 150 evaluation studies = 411 forward
passes (training studies additionally carry a backward pass).


| Run      | Blended time / study | Doubling check                               |
| -------- | -------------------- | -------------------------------------------- |
| AP       | 5.24 s               | —                                            |
| Sagittal | 5.24 s               | identical to AP, as expected (same 8 images) |
| Dual     | 11.37 s              | **2.17× AP** — consistent with 2× the images |


> **Audit caveat on step timing.** The code does **not** separately instrument the training
> step and the evaluation step; it records only whole-epoch wall time. The figures above are
> therefore *blended* (train + eval) per-study times, honestly labelled as such. Note that
> the `seconds_per_training_study` field written into `selection.yaml` (AP: 8.26 s) is
> **misleading and should not be quoted** — it divides *total* wall time, including all
> evaluation, by the *training* study count only, and so overstates the true training step
> cost by roughly 60%. It is left in place for artifact compatibility; this report does not
> use it.

Dual peaked at 8.08 GiB against a 24 GiB card, so the planned `num_frames: 6` fallback was
never needed — **dual saw the same 8 frames per view as the single-view runs**, and the
view comparison is not confounded by temporal resolution.

---



## 6. Full results on the 150-study validation set

Metrics are for the **early-stopping (selected) checkpoint** of each run. Class 0 = T012a
(unsuccessful, n=80), class 1 = T2b3 (successful, n=70). Brackets are stratified bootstrap
95% CIs, 10,000 replicates.

Two thresholds are reported throughout:

- **@ 0.5** — the natural decision threshold. *This is the number to compare against other
models.*
- **@ tuned** — the threshold that maximizes macro F1 **on these same 150 studies**. It is
reported for continuity with wave 1 and is **fitted on the data it scores**. It is not an
estimate of anything and must not be compared against other models' 0.5 numbers.



### 6.1 Threshold-free Metrics


| Run      | AUROC                    | AUPRC                |
| -------- | ------------------------ | -------------------- |
| AP       | 0.804 [0.730, 0.871]     | 0.779 [0.693, 0.865] |
| Sagittal | 0.845 [0.777, 0.905]     | 0.852 [0.790, 0.910] |
| Dual     | **0.857** [0.794, 0.914] | 0.840 [0.759, 0.915] |




### 6.2 AP — selected epoch 15 of 21, tuned threshold 0.9987

**Confusion matrix @ 0.5** (rows = actual, cols = predicted)


|                       | pred T012a | pred T2b3 |
| --------------------- | ---------- | --------- |
| **actual T012a** (80) | 47         | 33        |
| **actual T2b3** (70)  | 13         | 57        |


**Confusion matrix @ tuned (0.9987)**


|                       | pred T012a | pred T2b3 |
| --------------------- | ---------- | --------- |
| **actual T012a** (80) | 65         | 15        |
| **actual T2b3** (70)  | 20         | 50        |



| Metric                | @ 0.5                | @ tuned              |
| --------------------- | -------------------- | -------------------- |
| Accuracy              | 0.693 [0.620, 0.767] | 0.767 [0.693, 0.833] |
| Balanced accuracy     | 0.701 [0.629, 0.771] | 0.763 [0.692, 0.830] |
| Macro F1              | 0.692 [0.617, 0.765] | 0.764 [0.692, 0.832] |
| **T012a** precision   | 0.783                | 0.765                |
| **T012a** recall      | 0.588 [0.475, 0.688] | 0.812 [0.725, 0.900] |
| **T012a** F1          | 0.671                | 0.788                |
| **T012a** specificity | 0.814 [0.714, 0.900] | 0.714 [0.600, 0.814] |
| **T2b3** precision    | 0.633                | 0.769                |
| **T2b3** recall       | 0.814 [0.714, 0.900] | 0.714 [0.600, 0.814] |
| **T2b3** F1           | 0.713                | 0.741                |
| **T2b3** specificity  | 0.588 [0.475, 0.688] | 0.812 [0.725, 0.900] |
| Brier                 | 0.287                | —                    |
| ECE                   | 0.298                | —                    |


At 0.5 the model predicts positive for **90 of 150** studies against a true base rate of 70
— it over-calls success, missing 33 of 80 failures (T012a recall 0.588). See §7.

### 6.3 Sagittal — selected epoch 18 of 24, tuned threshold 0.5126

**Confusion matrix @ 0.5**


|                       | pred T012a | pred T2b3 |
| --------------------- | ---------- | --------- |
| **actual T012a** (80) | 60         | 20        |
| **actual T2b3** (70)  | 14         | 56        |


**Confusion matrix @ tuned (0.5126)**


|                       | pred T012a | pred T2b3 |
| --------------------- | ---------- | --------- |
| **actual T012a** (80) | 62         | 18        |
| **actual T2b3** (70)  | 14         | 56        |



| Metric                | @ 0.5                | @ tuned              |
| --------------------- | -------------------- | -------------------- |
| Accuracy              | 0.773 [0.707, 0.840] | 0.787 [0.720, 0.853] |
| Balanced accuracy     | 0.775 [0.707, 0.841] | 0.788 [0.720, 0.852] |
| Macro F1              | 0.773 [0.706, 0.840] | 0.786 [0.719, 0.852] |
| **T012a** precision   | 0.811                | 0.816                |
| **T012a** recall      | 0.750 [0.650, 0.838] | 0.775 [0.675, 0.863] |
| **T012a** F1          | 0.779                | 0.795                |
| **T012a** specificity | 0.800 [0.700, 0.886] | 0.800 [0.700, 0.886] |
| **T2b3** precision    | 0.737                | 0.757                |
| **T2b3** recall       | 0.800 [0.700, 0.886] | 0.800 [0.700, 0.886] |
| **T2b3** F1           | 0.767                | 0.778                |
| **T2b3** specificity  | 0.750 [0.650, 0.838] | 0.775 [0.675, 0.863] |
| Brier                 | 0.198                | —                    |
| ECE                   | 0.185                | —                    |


The tuned threshold (0.513) is essentially 0.5, so the two columns barely differ. This run
is the best-behaved of the three: its probabilities mean something.

### 6.4 Dual — selected epoch 15 of 21, tuned threshold 0.8395

**Confusion matrix @ 0.5**


|                       | pred T012a | pred T2b3 |
| --------------------- | ---------- | --------- |
| **actual T012a** (80) | 59         | 21        |
| **actual T2b3** (70)  | 14         | 56        |


**Confusion matrix @ tuned (0.8395)**


|                       | pred T012a | pred T2b3 |
| --------------------- | ---------- | --------- |
| **actual T012a** (80) | 66         | 14        |
| **actual T2b3** (70)  | 16         | 54        |



| Metric                | @ 0.5                | @ tuned              |
| --------------------- | -------------------- | -------------------- |
| Accuracy              | 0.767 [0.700, 0.833] | 0.800 [0.733, 0.860] |
| Balanced accuracy     | 0.769 [0.700, 0.835] | 0.798 [0.732, 0.861] |
| Macro F1              | 0.767 [0.699, 0.833] | 0.799 [0.732, 0.860] |
| **T012a** precision   | 0.808                | 0.805                |
| **T012a** recall      | 0.738 [0.637, 0.825] | 0.825 [0.738, 0.900] |
| **T012a** F1          | 0.771                | 0.815                |
| **T012a** specificity | 0.800 [0.700, 0.886] | 0.771 [0.671, 0.871] |
| **T2b3** precision    | 0.727                | 0.794                |
| **T2b3** recall       | 0.800 [0.700, 0.886] | 0.771 [0.671, 0.871] |
| **T2b3** F1           | 0.762                | 0.783                |
| **T2b3** specificity  | 0.738 [0.637, 0.825] | 0.825 [0.738, 0.900] |
| Brier                 | 0.195                | —                    |
| ECE                   | 0.198                | —                    |




### 6.5 Per-epoch val AUROC, and whether the selected epoch is a spike

Plan §9.2 required this check: with 150 studies, epoch-to-epoch AUROC wobble is comparable
to the real signal, so a single argmax epoch can overfit the selection set.

```
AP    (sel. ep15)  .592 .650 .701 .528 .708 .707 .724 .695 .723 .743 .759 .777 .799 .757
                   [.804] .767 .780 .769 .785 .769 .772
sag   (sel. ep18)  .583 .602 .627 .674 .698 .742 .757 .754 .802 .812 .816 .825 .824 .821
                   .819 .821 .838 [.845] .839 .839 .837 .837 .837 .837
dual  (sel. ep15)  .589 .662 .641 .641 .693 .736 .747 .773 .796 .822 .839 .838 .854 .835
                   [.857] .843 .846 .832 .836 .830 .830
```


| Run      | Argmax AUROC | Mean of last 5 epochs | Gap        |
| -------- | ------------ | --------------------- | ---------- |
| AP       | 0.804        | 0.775                 | **+0.029** |
| Sagittal | 0.845        | 0.837                 | +0.008     |
| Dual     | 0.857        | 0.835                 | **+0.022** |


**Sagittal's selected epoch is a plateau, not a spike** — its argmax sits 0.008 above its
own tail, so 0.845 is a trustworthy number. **AP and dual are mildly optimistic**: their
argmax sits ~0.02–0.03 above their plateaus. Their headline AUROCs should be read as the
top of a noisy band, not a stable capability. Deflating each run to its last-5 mean
(AP 0.775, sag 0.837, dual 0.835) leaves the ordering *and* the "indistinguishable"
conclusion unchanged.

---



## 7. Calibration: the probabilities are poor, especially AP


| Run      | Tuned threshold | Brier | ECE   | Predicted positive @ 0.5 (true: 70) |
| -------- | --------------- | ----- | ----- | ----------------------------------- |
| AP       | **0.9987**      | 0.287 | 0.298 | **90**                              |
| Sagittal | 0.5126          | 0.198 | 0.185 | 76                                  |
| Dual     | 0.8395          | 0.195 | 0.198 | 77                                  |


**AP's sigmoid saturated.** Its macro-F1-maximizing threshold is 0.9987 — meaning the model
pushed nearly all probability mass to the top of the range, and only by demanding
`P > 0.9987` can the classes be separated at all. This is the same probability-collapse
pathology wave 1 documented, and it is why AP's accuracy (0.693) badly lags its AUROC
(0.804): **the ranking is fine, the calibration is broken.** Moving AP to its tuned
threshold recovers 7 points of accuracy (0.693 → 0.767) purely by fixing the cut point.

This is the concrete vindication of the AUROC monitor. Had this wave early-stopped on macro
F1 @ 0.5, AP's selection signal would have been tracking sigmoid drift, not discrimination.

Sagittal and dual are meaningfully better calibrated (Brier ~0.20, ECE ~0.19), but none of
the three are clinically usable as probabilities. **Fixing this properly requires a split
that is neither train nor eval, which this protocol does not have** — temperature scaling on
the val set would just be another quantity fitted on the data it is scored against. Real
calibration needs a three-way split or k-fold CV.

---



## 8. Numerical audit

Every number in this report was **recomputed independently** from the 150-row per-study
score CSVs with fresh scikit-learn calls, and diffed against the values the training code
wrote. Result: **0 discrepancies.**


| Check                                                                                                                                                                                  | Result                                                                                |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| All point metrics (AUROC, AUPRC, F1, accuracy, balanced accuracy, per-class precision/recall/F1/specificity, at both thresholds, all 3 runs) recomputed from `scores.csv` vs. reported | **exact match** (Δ = 0.0)                                                             |
| Confusion-matrix cells sum to n                                                                                                                                                        | 150 for all 3 runs, both thresholds                                                   |
| Val class balance identical across runs                                                                                                                                                | 80 neg / 70 pos in all 3                                                              |
| `best_epoch` in `selection.yaml` == argmax of val AUROC in `history.json`                                                                                                              | matches (AP 15, sag 18, dual 15)                                                      |
| Reported AUROC == the selected epoch's logged AUROC                                                                                                                                    | exact match                                                                           |
| Bootstrap CIs bracket their point estimates                                                                                                                                            | all pass                                                                              |
| Brier / ECE recomputed vs. `history.json` at selected epoch                                                                                                                            | exact match                                                                           |
| Trainable parameter count identical across the 3 runs                                                                                                                                  | 29,810,177 in all 3                                                                   |
| Parameter breakdown recomputed from the saved checkpoints                                                                                                                              | 29,802,496 LoRA + 7,681 head = 29,810,177 (exact)                                     |
| **Checkpoint-reload fidelity:** train-time `val_`* artifacts vs. `final_*` re-scored from the reloaded adapter+head                                                                    | **bit-for-bit identical**, max per-study probability delta **0.0e+00** (AP, sagittal) |


That last check matters and is worth stating plainly. It proves the saved adapter and head
reload exactly, and it means the training-time artifacts *are* the final scored artifacts.
**Dual therefore needs no** `--phase final` **rerun** — its `val_seed14207_`* artifacts are its
final numbers, computed from the same checkpoint by the same code path that AP and sagittal
were independently confirmed against. (Dual's `--phase final` pass was deliberately stopped;
it would have recomputed identical values.)

**Specificity definition, verified:** `class_0_specificity` is the true-negative rate when
class 0 is treated as positive — i.e. the fraction of *actual class 1* studies correctly
identified — and is therefore numerically equal to class 1's recall (and vice versa). This
is correct for the binary case, and the mirror-image pairs visible in §6 are expected, not a
copy-paste error.

**Two caveats found and disclosed, neither affecting any reported number:**

1. `seconds_per_training_study` in `selection.yaml` conflates evaluation time into a
  "training" rate and overstates it by ~60%. Not used in this report (see §5).
2. `environment.yaml` records `git_commit: 3664974`, which is the repository HEAD at
  *artifact-write* time, not necessarily the code that ran. That commit
   (`modified plotting of barplot based on view`) touches only
   `plotting_functions/generate_frame_count_barplot.py` and landed while the runs were in
   flight. The training code that produced these results is `2812b6e`, unmodified
   throughout.

---



## 9. What this experiment establishes, and what it does not

**Establishes:**

- MedGemma 1.5 4B, QLoRA-adapted with a frozen vision tower, reaches **0.80–0.86 val AUROC**
on binary mTICI from DSA — clearly better than chance, and in the same territory as
VideoMAE sagittal.
- **Fine-tuning does all of the work.** Zero-shot MedGemma has no usable signal on this task
(§1.1): the selected zero-shot variant scores AUROC 0.403, *below* chance. QLoRA moves the
model from nothing to 0.80–0.86.
- It is **worse than both purpose-built model families.** CVFSNet fusion (0.928) and
VideoMAE AP (0.925) both lie outside the 95% CI of *every* MedGemma run, including
MedGemma's best (dual, CI upper 0.914).
- Its **probabilities are unreliable**, AP's catastrophically so — and this, not ranking
ability, is where most of the gap to the baselines lives.
- Dual-view (16 images, 4,096 vision tokens) fits in 8.1 GiB and costs 2.17× AP per study.

**Does not establish:**

- **Any ranking among the three views.** The gaps are inside the noise. Do not report
"dual is best" or "sagittal beats AP."
- Anything about held-out performance. There is no test set.
- That the frozen vision tower is *the* bottleneck — that is a hypothesis (§10), not a
finding.

---



## 10. Highest-value next steps

Ordered by expected value per GPU-hour. Items 1 and 2 are the ones that would actually
change a conclusion.

1. **A second seed** — the cheapest thing that would let any view comparison mean anything.
  Wave 1's four seeds spread 0.742–0.868 AUROC (sd 0.052). One seed gives no error bar
   beyond the bootstrap, and the current view "ordering" rests entirely on differences
   smaller than the seed-to-seed spread. **Nothing else on this list is interpretable until
   this is done.**
2. **Unfreeze the multimodal projector, then the vision tower.** The most likely bottleneck
  by a distance, and the natural explanation for why a 4B multimodal LLM loses to a much
   smaller video model. MedGemma's SigLIP tower was pretrained on radiology *stills*; DSA is
   a temporal contrast-flow modality far outside that distribution, so the language adapters
   are being asked to repair a representation they cannot reach.
   `target_scope: all_linear_language_projector` is already implemented — a one-line config
   change. (Note: PEFT matches `target_modules` by suffix, so bare leaf names like `q_proj`
   would silently adapt the tower too; targets are resolved to fully-qualified paths for
   exactly this reason, and an assert enforces the freeze. Unfreezing must relax that assert
   deliberately.)
3. **Free AP + sagittal ensemble.** Average the two single-view runs' val probabilities.
  Costs zero GPU time and gives a second "dual" number. If the 16-image model does not beat
   the ensemble, those extra 4,096 vision tokens are not buying cross-view reasoning.
4. **Frozen linear probe.** Never run. Answers "do the adapters help at all, or is MedGemma's
  frozen feature space already linearly separable?" Features can be cached once.
5. **Temporal resolution.** 8 frames trilinearly resampled from the full series may blur the
  contrast-flow events that *define* mTICI. 16 frames on AP is the most clinically motivated
   knob here — and dual's 8.08 GiB peak proves the memory headroom exists.
6. **Head pooling.** The head reads the last prompt token; mean-pooling over the image-token
  span is a small, local change and a plausible improvement.
7. **k-fold CV over all 411 studies.** The real fix for the elephant in this report. Every
  number in this repository, for every model family, is select-on-eval. A 5-fold CV would
   cost ~5× and would produce the first defensible number the project has had.

---



## Appendix: artifacts

Per run, under `output_runs_lightning/medgemma_qlora_val_{ap,sag,dual}/`:


| File                                       | Contents                                                            |
| ------------------------------------------ | ------------------------------------------------------------------- |
| `val_seed14207_scores.csv`                 | per-study logit, probability, true and predicted label (150 rows)   |
| `val_seed14207_metrics.yaml`               | all metrics at both thresholds + bootstrap CIs                      |
| `val_seed14207_confusion_matrix.png`       | confusion matrix at the tuned threshold                             |
| `seed_14207/history.json`                  | per-epoch train/val loss, AUROC, F1, threshold, calibration, timing |
| `seed_14207/selection.yaml`                | selected epoch, monitor, protocol, parameter counts, LoRA targets   |
| `seed_14207/adapter/`, `head.pt`           | the selected checkpoint                                             |
| `resolved_config.yaml`, `environment.yaml` | full config and environment as run                                  |


AP and sagittal additionally carry `final_seed14207_*` (identical values — see §8).
Split manifest: `output_runs_lightning/medgemma_full_train_val_split.json` (261/150/150).

**Cross-model metrics tables of record** (where these runs' headline numbers now live,
alongside CVFSNet and VideoMAE):


| File                                                           | Contents                                                                                                      |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `Assets/confusion_matrix_metrics_cvfsnet_binary_runs_mod3.csv` | CVFSNet + MedGemma QLoRA (3 runs) + MedGemma zero-shot (2 variants), with tp/tn/fp/fn and `split`/`n` columns |
| `Assets/confusion_matrix_metrics_videomae_binary_runs.csv`     | VideoMAE                                                                                                      |
| `Assets/metrics_comparison_table.csv`                          | generated Model x View table, via `python -m amticis_training.build_metrics_comparison_table`                 |


The MedGemma rows there are at threshold 0.5 (not the tuned threshold), so they are
like-for-like with the other families. The zero-shot rows carry `split=tuning, n=53` and
must not be read as val-split results.