# MedGemma QLoRA Wave 2 — AP / Sagittal / Dual on the Val Protocol

**Status: executing (2026-07-12).** Sections 5 and 6 are done and committed (`2812b6e`).
The smoke test passed on all three view configs and the three full runs of section 7 are
training. Remaining: `--phase final` per run, then the section-8 report.

Two things the smoke test settled, both of which the plan flagged as open risks:

- **Dual view fits comfortably.** Peak 8.08 GiB at 16 images/study against a 24 GiB card,
  so the `num_frames: 6` fallback in §6.4 is *not* needed and dual sees the same 8 frames
  per view as the single-view runs. The comparison is clean.
- **Sagittal routing is correct.** The sagittal run resolves `*_S*.nii.gz` and the AP run
  `*_C*.nii.gz`, verified both by literal path and by pixel content (the two views' tensors
  differ; each single-view config matches the corresponding view inside the dual prompt).
  This was §9.6's worry — a silent view bug would have been indistinguishable from
  "sagittal is just harder."

Measured epoch times (261 train + 150 eval, three runs sharing the box): AP ~34 min,
sagittal ~35 min, dual ~74 min — close to the §4 estimates, so the §4 wall-clock still
holds.

---

## 1. Context

Wave 1 (`QLORA_AP_TUNING_REPORT.md`) trained MedGemma QLoRA on a 208-study development
set, selected the epoch and threshold on a 53-study tuning split, and left the 150-study
`val` split locked and unread. That protocol is statistically clean but has two costs: it
discards 20% of the training data, and it produces numbers that **cannot be compared** to
any other experiment in this repository, because VideoMAE and CVFSNet both train on all
261 studies and select the best epoch on the 150-study `val` set.

This wave puts MedGemma on that same protocol so the comparison is apples-to-apples, and
extends it from AP-only to all three view configurations.

| Model | Accuracy | Macro F1 | AUROC |
|---|---:|---:|---:|
| VideoMAE AP | 0.827 | 0.826 | 0.925 |
| VideoMAE sagittal | 0.780 | 0.779 | 0.845 |
| VideoMAE dual | 0.853 | 0.853 | 0.895 |
| CVFSNet `binary_cor_v3` (fused head) | 0.867 | 0.867 | 0.895 |
| CVFSNet `binary_fusion_v3` (fused head) | 0.853 | 0.853 | 0.905 |
| **MedGemma QLoRA AP** | — | — | — |
| **MedGemma QLoRA sagittal** | — | — | — |
| **MedGemma QLoRA dual** | — | — | — |

Filling in those last three rows is the deliverable.

**Accepted bias, state it in the report:** early stopping and reporting both happen on the
same 150-study `val` set. There is no test set. Every number this wave produces is
optimistically biased by best-epoch-on-eval selection — exactly as the VideoMAE and
CVFSNet rows above already are. That shared bias is the point (it makes the rows
comparable), but none of these are estimates of held-out performance.

---

## 2. Design

Three runs, one seed (`14207`), one GPU and one terminal each. Hyperparameters are
**unchanged** from wave 1 (`medgemma_binary/config_qlora_ap.yaml`): QLoRA 4-bit NF4,
r=16 / α=16 / dropout 0.05, `all_linear_language` targets (vision tower frozen), adapter
LR 2e-4, head LR 1e-3, micro-batch 1 × 8 gradient accumulation, cosine schedule with 3%
warmup, grad clip 0.3, 25 max epochs, patience 6, prompt `clinical_v1`, scaling
`percentile_1_99`, 8 frames, 896×896.

| GPU | Run name | `views` | Images/study | Train | Eval |
|---:|---|---|---:|---:|---:|
| 0 | `medgemma_qlora_val_ap` | `[AP]` | 8 | 261 | 150 |
| 1 | `medgemma_qlora_val_sag` | `[sagittal]` | 8 | 261 | 150 |
| 2 | `medgemma_qlora_val_dual` | `[AP, sagittal]` | 16 | 261 | 150 |

**Run 3 is one datapoint per study** — a single prompt containing `AP VIEW / FRAME 1..8`
followed by `SAGITTAL VIEW / FRAME 1..8`, 16 images, one probability out. Nothing is
averaged, and there is no per-view expansion of the dataset. `build_messages`
(`qlora.py:49`) already loops over views and emits those markers, and
`MedGemmaBinaryClassifier.expected_images` (`qlora.py:194`) already computes
`num_frames × len(views)` and asserts the processor returns that many `pixel_values`.
**Run 3 therefore needs no model or dataset code — only config.**

**Early stopping monitors val AUROC (max), patience 6.** AUROC is the right monitor here:
wave 1 showed the sigmoid scale is an artifact of the seed — seed 2718 never emitted a
probability above 0.307 and was degenerate at a 0.5 threshold (balanced accuracy exactly
0.500) while still scoring AUROC 0.742. Any thresholded selection metric tracks that
probability collapse rather than discrimination. AUROC is threshold-free and was the only
seed-stable quantity in wave 1 (sd 0.052 vs 0.213 for macro F1 @ 0.5).

**Reporting.** AUROC is the headline. Also report macro F1 @ 0.5, balanced accuracy,
AUPRC, per-class precision/recall/F1/specificity, Brier, ECE, and stratified bootstrap 95%
CIs (10,000 replicates) — all already implemented in `common.summarize_rows`. Macro F1 at a
val-tuned threshold is also reported for continuity with wave 1, but must be labelled as
fitted-on-the-same-set, not an estimate of anything.

---

## 3. Key facts about the existing code

Established by reading the source; trust these rather than re-deriving them.

- **Splits.** `MedGemmaData.final` is exactly the repo's standard 150-study `val` split;
  `development` + `tuning` is exactly the standard 261-study `train` split, cut 80/20 by
  `selection.split_seed: 14207`. Verified against
  `output_runs_lightning/medgemma_internal_split.json` (208 + 53 + 150).
- **Views are config-only**, set in **two places that are cross-checked**:
  `data.pipeline_overrides.views.active` and `input.views`. `data.py:47` raises if they
  disagree. File resolution is by marker substitution on the coronal base name
  (`AP → _C`, `sagittal → _S`, in `amticis_pipeline/config.yaml`). The `_S` NIfTI files
  exist on disk for all 411 studies.
- **GPU selection is via `CUDA_VISIBLE_DEVICES` only.** `qlora.py:673` hard-codes
  `torch.device("cuda", 0)`. There is no flag and no config key.
- **AUROC is already computed every epoch** (`qlora.py:458`, via
  `summarize_rows → _point_metrics → roc_auc_score`). Adding an AUROC monitor is a
  comparison change, not a metrics change.
- **`evaluation.threshold: 0.5` in the YAML is dead** — `qlora.py` hard-codes `0.5` at
  lines 309, 425, 429. Don't be misled by it.
- **Batch size is structurally 1.** The head reads `last_hidden_state[:, -1, :]`
  (`qlora.py:213`), which is only the last prompt token because there is no padding. Do
  not "optimize" eval by batching without padding-aware indexing.
- **Wave-1 artifacts must keep working.** Four seeds live under
  `output_runs_lightning/medgemma_qlora_binary_tici_ap/`, and no `final_*` artifacts exist
  (the locked split has never been consumed). Do not overwrite that run directory, and do
  not consume that locked split.

---

## 4. Cost

Measured wave-1 rates on one RTX 3090: 5.25 s/study training step, 3.61 s/study eval step,
peak 6.1 GiB at 8 images.

| Run | Epoch (261 train + 150 eval) | 25 epochs | Realistic (early stop ~ep 16) |
|---|---:|---:|---:|
| AP | ~32 min | ~13 h | ~9 h |
| Sagittal | ~32 min | ~13 h | ~9 h |
| Dual (16 images) | ~64 min | ~27 h | ~18 h |

All three run concurrently on three GPUs, so the wave is gated by dual: **~18–27 h wall
clock**.

---

## 5. Code changes

All additive. `selection.protocol` and `training.monitor` both default to the wave-1
behaviour, so the existing configs and artifacts are untouched.

### 5.1 `data.py` — split protocol

Add `selection.protocol`, defaulting to `internal` (the wave-1 behaviour), with a new
`full_train_val` mode. Three places in `MedGemmaData` need to change, and **the last two
are easy to miss** — they are the reason the smoke test exists.

- **`__init__` (lines 64–83).** Read `selection.protocol` and validate it. Under
  `full_train_val`, set `development` to all 261 `train` studies and `tuning` to the 150
  `val` names; `final` stays the 150 `val` names. Skip `make_internal_split` entirely — it
  cannot produce an empty tuning set (`sklearn.train_test_split(test_size=0.0)` raises,
  `common.py:157`).

  Assert disjointness over `{development, final}` **only**. `tuning` deliberately aliases
  `final` under this protocol, and the pairwise check in `assert_disjoint_studies` would
  otherwise flag the 150 studies as overlapping themselves and raise.

- **`labels_for` (lines 99–102)** looks names up in the **`train`** dataset only:
  ```python
  dataset = self.eval_module._datasets["train"]
  index = {name: position for position, name in enumerate(dataset.samples)}
  return [dataset.label_at(index[name]) for name in names]
  ```
  Under `full_train_val`, `self.tuning` holds **val** names, so `labels_for(data.tuning)`
  raises `KeyError`. It is called from `main()` whenever `--limit` is passed, i.e. **the
  smoke test crashes on it**. Fix: build the index over both the `train` and `val` datasets
  of `eval_module` and resolve each name against whichever contains it.

- **`loader` (lines 104–119)** hard-codes `"tuning" → (eval_module, "train", …)`. Under
  `full_train_val` the tuning names live in the **`val`** dataset, so the
  `ValueError("Split 'tuning' is not fully present in the train dataset.")` guard at the
  end of the method fires. Fix: pick the dataset key by protocol — `"train"` under
  `internal`, `"val"` under `full_train_val`. Leave `development` on `train_module`/`train`
  (augmented) and `final` on `eval_module`/`val` (deterministic) as they are.

- **Manifest.** `_reconcile_manifest` does an exact dict-equality check against
  `selection.manifest_path` and raises if it disagrees. A 261/150/150 manifest will not
  equal the existing 208/53/150 one, so the new configs **must** point
  `selection.manifest_path` at a new file
  (`output_runs_lightning/medgemma_full_train_val_split.json`). Do **not** relax the check —
  it is doing its job of stopping two experiments from silently using different splits.

### 5.2 `qlora.py` — configurable early-stopping monitor

`improved` is currently the hard-coded lexicographic tuple at `qlora.py:475`:
```python
improved = (epoch_f1, metrics["auroc"]) > (best["macro_f1"], best["auroc"])
```
Add `training.monitor`, defaulting to `f1_macro_then_auroc` (current behaviour), with a new
`auroc` mode that compares `metrics["auroc"]` alone. Validate the value and raise on
anything else. Record the chosen monitor in the `selection_metric` field of
`selection.yaml` (`qlora.py:499`, currently the hard-coded string
`"f1_macro_at_tuned_threshold_then_auroc"`) so the artifact is self-describing.

The tuned threshold is still computed and stored every epoch — it is needed for reporting
and by `--phase final` — it just no longer drives selection under `monitor: auroc`.

### 5.3 `qlora.py` — smoke phase must honour `--set`

`main()` calls `load_config(args.config, args.set)` (line 639, which applies the `--set`
overrides), and *then* the smoke branch at lines 650–653 force-writes
`max_epochs = 1`, `early_stopping_patience = 1`, `bootstrap_replicates = 100`. So
`--set training.max_epochs=3` is silently discarded. Reorder so the smoke defaults are
applied **first** and explicit `--set` overrides win. The 3-epoch smoke test depends on
this.

### 5.4 `qlora.py` — artifact naming

`write_artifacts` is called with `split_name=f"tuning_seed{seed}"` (line 516), and the
per-epoch `record` keys are prefixed `tuning/` (lines 453–464). Under this protocol that
split *is* `val`, and a file called `tuning_seed14207_scores.csv` full of val scores is a
trap for whoever reads the run directory in six months. Add `run.eval_split_label`
(default `tuning`, set to `val` in the new configs) and use it for both the artifact
prefix and the record-key prefix.

Leave the key names *inside* `selection.yaml` (`tuning_auroc`, etc.) alone —
`evaluate_final` reads `tuned_threshold` and `best_epoch` from it, and renaming buys
nothing.

### 5.5 New configs

`config_qlora_val_ap.yaml`, `config_qlora_val_sag.yaml`, `config_qlora_val_dual.yaml` —
each a copy of `config_qlora_ap.yaml` changing **only**:

```yaml
seed: 14207

run:
  name: medgemma_qlora_val_ap        # _sag | _dual
  eval_split_label: val

data:
  pipeline_overrides:
    views:
      active: [AP]                   # [sagittal] | [AP, sagittal]

input:
  views: [AP]                        # [sagittal] | [AP, sagittal] -- must match the above

training:
  monitor: auroc

selection:
  protocol: full_train_val
  manifest_path: output_runs_lightning/medgemma_full_train_val_split.json

wandb:
  group: e2_qlora_val_protocol
```

Everything else is inherited unchanged. Note `input.views` and
`data.pipeline_overrides.views.active` must agree or `data.py:47` raises.

### 5.6 Tests — `tests/test_medgemma_qlora.py`

**`tests/*` is gitignored**; the existing test files are tracked by force-add
(`git add -f`). New test files need the same, or they will silently vanish.

Extend, don't replace:

- `full_train_val` yields 261 development / 150 tuning / 150 final, and `development` is
  the full `train` list.
- `development` ∩ `final` is patient-disjoint.
- `internal` still yields 208 / 53 / 150 (regression guard on wave 1).
- `labels_for` resolves **val** names under `full_train_val` (this is the bug in 5.1).
- `monitor: auroc` selects the max-AUROC epoch even when a later epoch has a higher
  tuned-threshold macro F1.
- `expected_images` is 8 / 8 / **16** for the three configs, and a mismatch between
  `input.views` and `pipeline.views.active` still raises.

Run with the isolated environment:
```bash
.venv-medgemma/bin/python -m unittest tests.test_medgemma_qlora -v
```
(24 tests pass today; they run in ~0.1 s and need no GPU.)

---

## 6. Smoke test — run this before committing any GPU time

3 epochs, 8 training + 8 eval studies, balanced. The point is to exercise every code path
that differs from wave 1, not to produce numbers.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-medgemma/bin/python -m medgemma_binary.qlora \
    --config medgemma_binary/config_qlora_val_ap.yaml \
    --phase smoke --limit 8 --set training.max_epochs=3
# repeat for config_qlora_val_sag.yaml (GPU 1) and config_qlora_val_dual.yaml (GPU 2)
```

The full runs do not start until all of this holds:

1. **Split sizes** print as `development=261 tuning=150 final=150` **before** `--limit`
   truncates them (`main()` currently prints them after; move or duplicate the print), and
   the new manifest is written to `medgemma_full_train_val_split.json` with the wave-1
   manifest untouched on disk.
2. **View routing**: the AP run reads `*_C*.nii.gz`, the sagittal run reads `*_S*.nii.gz`,
   the dual run reads both. Log one resolved path per view and eyeball it.
3. **`expected_images`** is 8 / 8 / **16**, and the processor's `pixel_values` first
   dimension matches — the assert at `qlora.py:204` fires otherwise.
4. **Dual-view fits in 24 GiB.** AP-only peaked at 6.1 GiB with a ~2204-token sequence;
   dual doubles the vision tokens to 4096 (~4300-token sequence). Expect roughly 9–12 GiB.
   **This is the one genuine risk in the wave.** If it OOMs, the fallback is
   `input.num_frames: 6` for the dual run only — and the report must then state that dual
   saw fewer frames per view than the single-view runs, which weakens the comparison.
5. **Vision tower still frozen**: the assert at `qlora.py:173` passes and trainable
   parameters are 29,810,177 for all three runs. The count does not change with view count
   (the tower and projector are excluded), so a different number means something broke.
6. **Monitor is AUROC**: `selection.yaml` records `selection_metric: auroc`, and the epoch
   chosen across the 3 smoke epochs is the max-AUROC one, not the max-F1 one.
7. **Loss is finite** at every step and `train/loss` moves; artifacts land under
   `output_runs_lightning/medgemma_qlora_val_{ap,sag,dual}/` named `val_seed14207_*`, with
   no collision with the wave-1 run directory.
8. **Step time** is logged and is in the neighbourhood of the section-4 estimate.

---

## 7. Full runs

One terminal per GPU, all three concurrently:

```bash
# terminal 1
CUDA_VISIBLE_DEVICES=0 .venv-medgemma/bin/python -m medgemma_binary.qlora \
    --config medgemma_binary/config_qlora_val_ap.yaml --phase train --seed 14207 \
    2>&1 | tee logs/qlora_val_ap.log

# terminal 2
CUDA_VISIBLE_DEVICES=1 .venv-medgemma/bin/python -m medgemma_binary.qlora \
    --config medgemma_binary/config_qlora_val_sag.yaml --phase train --seed 14207 \
    2>&1 | tee logs/qlora_val_sag.log

# terminal 3
CUDA_VISIBLE_DEVICES=2 .venv-medgemma/bin/python -m medgemma_binary.qlora \
    --config medgemma_binary/config_qlora_val_dual.yaml --phase train --seed 14207 \
    2>&1 | tee logs/qlora_val_dual.log
```

Then, per run, regenerate the full artifact set (bootstrap CIs, confusion matrix,
per-study CSV) from the selected checkpoint at the frozen threshold:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-medgemma/bin/python -m medgemma_binary.qlora \
    --config medgemma_binary/config_qlora_val_ap.yaml --phase final --seed 14207
```

Under this protocol `--phase final` re-scores the **same** 150 studies the run
early-stopped on. It adds no new bias, but it is a reporting convenience, **not** an
independent evaluation, and the report must not present it as one.

---

## 8. Deliverable

`medgemma_binary/docs/QLORA_VAL_PROTOCOL_REPORT.md`, containing:

- The three runs' val AUROC with bootstrap CIs, macro F1 @ 0.5, balanced accuracy,
  calibration (Brier, ECE), and confusion matrices.
- Per-epoch AUROC curves, with the selected epoch marked — and an honest note on whether
  the argmax is a plateau or a spike (see improvement 2 below).
- The section-1 comparison table, filled in.
- An explicit statement that every row in that table, MedGemma and baselines alike, shares
  the same select-on-eval optimistic bias.

---

## 9. What to improve before training

1. **One seed gives no error bar.** Wave 1's four seeds spread 0.742–0.868 AUROC
   (sd 0.052) on 53 studies. On 150 studies the bootstrap CI will still be roughly
   ±0.06–0.08. **Any difference below ~0.08 AUROC between the three views is not a
   result.** If the three runs land inside that band, the honest conclusion is "the views
   are indistinguishable at this sample size," and a second seed becomes the
   highest-value follow-up. Decide that in advance, not after seeing a 0.03 gap.
2. **Val-AUROC epoch selection is itself noisy.** With 150 studies, epoch-to-epoch AUROC
   wobble is comparable to the real signal, so taking the single argmax epoch overfits the
   selection set. Cheap mitigation: the per-epoch AUROC is already logged — in the report,
   show the curve and the mean of the last 5 epochs alongside the argmax, and say so if
   the argmax is a spike.
3. **Wave 1 overfit hard**: train loss 0.133 vs eval loss 1.544 by the last epoch, a gap of
   1.4, while AUROC kept *improving*. Training on 261 instead of 208 helps slightly;
   nothing else in the recipe changes. Expect the same shape, and **do not early-stop on
   loss.**
4. **Class balance is fine** (train 128/133, val 80/70). No `pos_weight`, no weighted
   sampler. Leave `use_weighted_sampler: false`.
5. **Smoke the dual run first.** It is both the memory risk and the run you most want to
   succeed; find out in 10 minutes, not 18 hours.
6. **Sanity-check the sagittal path end-to-end.** No MedGemma experiment in this repo has
   ever *trained* on `_S` files — the zero-shot runs loaded both views into one prompt, and
   wave 1 was AP-only. A silent view-resolution bug would be indistinguishable from "the
   sagittal view is just harder."

---

## 10. What to investigate for performance improvement

Ordered by expected value per GPU-hour.

1. **Unfreeze the multimodal projector, then the vision tower.** This is the most likely
   bottleneck by a distance. MedGemma's SigLIP tower was pretrained on radiology stills and
   is frozen here; DSA is a temporal contrast-flow modality far outside that distribution,
   so the language adapters are being asked to repair a representation they cannot reach.
   `target_scope: all_linear_language_projector` is already implemented
   (`resolve_target_modules`, `qlora.py:67`) — this is a one-line config experiment.
   Note the wave-1 bring-up defect: PEFT matches `target_modules` **by suffix**, so bare
   leaf names like `q_proj` silently adapt the vision tower too. Targets are resolved to
   fully-qualified paths for exactly this reason, and there is an assert that the tower
   stays frozen. If you deliberately unfreeze it, that assert has to be relaxed
   deliberately, not deleted.
2. **Free AP + sagittal ensemble.** Average the two single-view runs' val probabilities.
   Costs zero GPU time and yields a second dual-view number to compare against the 16-image
   model. If the true dual model does not beat the ensemble, those extra 4096 vision tokens
   are not buying cross-view reasoning and the architecture is not earning its cost.
3. **Frozen linear probe (E1 in `BINARY_TICI_EXPERIMENT_PLAN.md`).** Never run. It answers
   "do the adapters actually help, or is MedGemma's frozen feature space already linearly
   separable?" — which decides whether QLoRA is even the right tool. Cheap: features can be
   cached once.
4. **Temporal resolution.** 8 frames trilinearly resampled from the full series may blur the
   contrast-flow events that *define* mTICI. Try 16 frames on the AP run (doubles vision
   tokens — measure memory before committing). This is the most clinically-motivated knob on
   the list.
5. **Head pooling.** The head reads the *last prompt token* (`qlora.py:213`). Mean-pooling
   over the image-token span is a plausible improvement and a small, local change.
6. **LoRA capacity.** r=16/α=16 → r=32/α=32. Cheap, but the model already memorizes 208
   studies, so more capacity is likelier to overfit than to help. Low priority.
7. **Calibration.** Wave 1's probabilities were uncalibrated and seed-dependent (ECE 0.268,
   Brier 0.261). Temperature scaling needs a split that is neither train nor eval — which
   this protocol does not have. If calibrated probabilities ever matter clinically, that
   requires going back to a three-way split or k-fold CV.
8. **The real fix for all of it: k-fold CV over all 411 studies.** Every number in this
   repository, for every model family, is select-on-eval. A 5-fold CV would cost ~5× but
   would produce the first defensible number the project has had. Worth planning once the
   view question is settled.
