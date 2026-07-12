# MedGemma QLoRA Binary TICI Experiment (E2, AP view)

This document records the implementation of experiment E2 from
[BINARY_TICI_EXPERIMENT_PLAN.md](BINARY_TICI_EXPERIMENT_PLAN.md), restricted to
the AP (coronal) view. It covers the hyperparameters, where they come from, two
implementation defects found during bring-up, and the measured cost of a run.

Results from the four-seed training wave are in
[QLORA_AP_TUNING_REPORT.md](QLORA_AP_TUNING_REPORT.md).

Binary labels are unchanged: class 0 is `T012a` (mTICI T0/T1/T2A), class 1 is
`T2b3` (mTICI T2B/T3).

## 1. Files

- [`qlora.py`](../qlora.py) — model, training loop, evaluation, CLI
- [`data.py`](../data.py) — development/tuning/final study sets over the shared pipeline
- [`config_qlora_ap.yaml`](../config_qlora_ap.yaml) — the resolved experiment configuration
- [`tests/test_medgemma_qlora.py`](../../tests/test_medgemma_qlora.py) — 24 tests

## 2. Data

The experiment reuses the repository's existing `AmTICISDataModule` and its
`ResizeView` trilinear resampling. Nothing about frame count, image size, or
view selection is hard-coded; all of it flows from the YAML.

```text
raw AP scan (H, W, T)
  -> permute to (1, 1, T, H, W)
  -> one trilinear interpolation over time and space (align_corners: true)
  -> (1, 8, 896, 896)
  -> per-scan percentile (1, 99) intensity scaling
  -> 8 grayscale frames repeated to RGB
  -> AutoProcessor -> pixel_values (8, 3, 896, 896)
```

| Partition | Class 0 | Class 1 | Total | Use |
|---|---:|---:|---:|---|
| Development | 102 | 106 | 208 | QLoRA training |
| Tuning | 26 | 27 | 53 | Checkpoint, epoch, and threshold selection |
| Locked final | 80 | 70 | 150 | Read exactly once, by `--phase final` |

The split is byte-identical to the one the zero-shot experiments used: it is
regenerated deterministically from `selection.split_seed: 14207` and then
reconciled against `output_runs_lightning/medgemma_internal_split.json`, which
raises if it ever disagrees. `split_seed` is deliberately independent of the
training `seed`, so the four training seeds all see the same partition. Patient
IDs are asserted disjoint across all three partitions.

### Augmentation

Per section 5.3 of the plan, only mild frame-consistent geometry is applied, and
only to the development set:

```yaml
- {name: RandomRotation, params: {degrees: [-10, 10], fill: 192, p: 0.25}}
- {name: Crop,           params: {crop: [0.1, 0.1, 0.2, 0.1], p: 0.5}}
- {name: Resize,         params: {}}   # t / visual injected from data.*
```

`torchvision` samples one angle per call and applies it to the whole
`(C, T, H, W)` clip, so no artificial temporal motion is introduced. No
intensity normalization runs in the pipeline: percentile scaling happens once
per scan in `scan_to_rgb_frames`, so it stays shared across a scan's frames.
Tuning and final studies are served by a second, deterministic data module, so
model selection never sees a randomly augmented scan.

## 3. Why 896 x 896 and not 256 x 256

MedGemma's vision encoder is the 400M SigLIP variant with a **fixed 896 x 896
input** producing **256 tokens per image** (`mm_tokens_per_image: 256` in the
model config; MedGemma and MedGemma 1.5 technical reports). The processor
resizes whatever it is handed back to 896 x 896, so feeding 256 x 256 scans
would simply upsample them again — discarding detail for no memory saving and no
speedup. Every published MedGemma result therefore uses 896.

At AP-only with 8 frames this costs 8 x 256 = 2048 vision tokens, and a study
encodes to a 2204-token sequence. That is comfortable: peak GPU use is 6.1 GiB.

## 4. Hyperparameters and their provenance

The QLoRA recipe follows Google's official MedGemma fine-tuning notebook
(`Google-Health/medgemma`, `notebooks/fine_tune_with_hugging_face.ipynb`), which
is itself the QLoRA-paper recipe.

| Parameter | Value | Source |
|---|---|---|
| Quantization | 4-bit NF4, double quant, BF16 compute | Official notebook |
| LoRA rank / alpha / dropout | 16 / 16 / 0.05 | Official notebook |
| LoRA targets | all Linear layers in the language model | Official notebook (`all-linear`), narrowed — see below |
| Adapter LR | 2e-4 | Official notebook (QLoRA paper) |
| Max grad norm | 0.3 | Official notebook (QLoRA paper) |
| Warmup ratio | 0.03 | Official notebook (QLoRA paper) |
| Precision / checkpointing | BF16, gradient checkpointing on | Official notebook |
| Head LR | 1e-3 | Plan section 7.3 |
| Head dropout | 0.2 | Plan section 7.3 |
| Weight decay | 0.01 | Plan section 7.3 |
| Effective batch | 1 x 8 accumulation | Plan section 7.3 |
| Max epochs / patience | 25 / 6 | Plan section 7.3 |
| Scheduler | cosine | Plan section 7.3 |

Three deliberate deviations from the notebook:

1. **Targets are restricted to the language model.** The notebook's `all-linear`
   would also adapt the vision tower. Plan section 7.3 asks to start with the
   vision encoder frozen, and the MedGemma 1.5 report states that MedSigLIP was
   itself frozen while the language decoder was trained. This resolves to 238
   modules (34 layers x {q,k,v,o,gate,up,down}_proj).
2. **`modules_to_save` (`lm_head`, `embed_tokens`) is dropped.** The notebook
   performs generative SFT; this experiment reads the final hidden state at the
   last prompt token through a `LayerNorm -> Dropout -> Linear(2560, 1)` head
   trained with BCE-with-logits (plan section 6), so the language-model head is
   unused.
3. **Micro-batch size 1.** One study is already 8 images; with batch size 1 the
   final sequence position is unambiguously the last prompt token, with no
   padding arithmetic.

Trainable parameters: **29,810,177** of 1,615,104,881 (1.85%).

## 5. Two defects found during bring-up

### 5.1 LoRA was silently adapting the frozen vision tower

PEFT matches `target_modules` **by suffix**. Passing bare leaf names — the
obvious reading of the plan's `target_modules: [q_proj, k_proj, v_proj, o_proj]`
— also matches the SigLIP vision encoder's `q_proj`/`k_proj`/`v_proj`, so
adapters were injected into the encoder the plan requires to stay frozen. The
tower reported 162 parameters requiring gradients.

The fix is to resolve target modules to **fully-qualified paths**
(`model.language_model.layers.N...`). `MedGemmaBinaryClassifier` now asserts the
tower is frozen after `get_peft_model` and refuses to start otherwise, and
`tests/test_medgemma_qlora.py` regression-tests it against a fake module tree
that reproduces the real naming.

### 5.2 The frozen tower was being checkpointed and back-propagated

Because of 5.1, every training step ran a backward pass and a
gradient-checkpoint recompute through 8 x 4096 patch tokens x 27 encoder layers.
With the tower genuinely frozen its activations never enter the backward graph,
so checkpointing it only pays for a recompute nothing consumes; it is now
explicitly disabled on the tower.

Measured effect of the two fixes together, per training study on one RTX 3090:

| | Step time | Peak GPU |
|---|---:|---:|
| Before | 17.8 s | 11.4 GiB |
| After | 5.3 s | 6.1 GiB |

## 6. Cost

Measured on one RTX 3090 at `num_frames: 8`, AP only:

| Quantity | Value |
|---|---:|
| Training step | 5.25 s/study |
| Evaluation step | 3.61 s/study |
| One epoch (208 train + 53 tuning) | ~21 min |
| 25 epochs | ~9 h |
| Peak GPU memory | 6.1 GiB |

The four seeds (14207, 42, 1337, 2718) run concurrently, one per GPU, so a full
four-seed wave is ~9 h of wall-clock rather than ~36 h. This satisfies the
plan's requirement of at least three seeds per trainable strategy in a single
pass. Roughly 3.6 s of each 5.25 s step is the frozen vision tower, which is
irreducible while augmentation is on (augmented images change every epoch, so
image features cannot be cached).

## 7. Running it

```bash
# Fast end-to-end check (1 epoch, 8 development + 8 tuning studies)
CUDA_VISIBLE_DEVICES=0 .venv-medgemma/bin/python -m medgemma_binary.qlora \
    --phase smoke --limit 8

# One seed per GPU
CUDA_VISIBLE_DEVICES=0 .venv-medgemma/bin/python -m medgemma_binary.qlora \
    --phase train --seed 14207

# Locked 150-study evaluation, using the epoch and threshold chosen on tuning
CUDA_VISIBLE_DEVICES=0 .venv-medgemma/bin/python -m medgemma_binary.qlora \
    --phase final --seed 14207
```

Runs are tracked in Weights & Biases under project `medgemma-binary-tici`, group
`e2_qlora_ap`.

## 8. Selection protocol

Everything is selected on the 53-study tuning split and committed to
`selection.yaml` before the final split is touched:

- **Epoch**: highest tuning macro F1, then AUROC. Early stopping patience 6.
- **Threshold**: the tuning-split threshold maximizing macro F1. Final metrics
  are reported both at 0.5 and at this threshold.

`--phase final` loads the saved adapter and head, evaluates the 150 locked
studies once, and exports one row per study
(`name,true_label,probability_t2b3,predicted_label,threshold,experiment,seed`)
plus macro F1, accuracy, balanced accuracy, AUROC, AUPRC, per-class
precision/recall/F1/specificity, Brier score, expected calibration error, and
stratified bootstrap 95% confidence intervals (10,000 replicates).

## 9. Limitations

- MedGemma is not clinically validated for reperfusion grading, and its
  high-dimensional radiology training (ordered CT/MRI slices) is not equivalent
  to temporal DSA angiography.
- 208 training studies is small for 29.8M trainable parameters; the tuning split
  is only 53 studies, so selection is noisy and the tuning point estimate is
  optimistically biased.
- This run uses the AP view only. The plan's primary configuration is dual-view.
- Trilinear temporal interpolation standardizes sequence length but may blur
  short contrast-flow events.
- Zero-shot MedGemma was at or below chance on this tuning split (macro F1 0.440,
  AUROC 0.403), so E2 is being trained on top of a backbone with no demonstrated
  DSA reperfusion discrimination.
