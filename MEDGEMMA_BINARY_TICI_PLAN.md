# MedGemma 1.5 4B Plan for Binary TICI Classification

## 1. Objective

Evaluate MedGemma 1.5 4B for binary mTICI classification on the AmTICIS DSA
dataset and compare four adaptation strategies under the same data split,
preprocessing, prompt, classification head, metrics, and random seeds.

| Class | Label | Original grades |
|---|---:|---|
| T012a | 0 | T0, T1, T2A |
| T2b3 | 1 | T2B, T3 |

The core comparison will include:

1. Zero-shot MedGemma.
2. Frozen MedGemma with a trainable linear classification head.
3. LoRA/QLoRA fine-tuning with a trainable classification head.
4. Full fine-tuning with a trainable classification head.

This document is a plan only. Implementation and model download are outside
its current scope.

## 2. Model and Access

Use the gated Hugging Face checkpoint:

```text
google/medgemma-1.5-4b-it
```

MedGemma 1.5 does not provide a separate 3D checkpoint. Its high-dimensional
imaging workflow supplies an ordered series of 2D images to the multimodal
model. For DSA, the images will be ordered temporal frames rather than CT or
MRI slices.

Before implementation:

1. Confirm that the Hugging Face account associated with `HF_TOKEN` has
   accepted the Health AI Developer Foundations terms for the model.
2. Download the model into the Hugging Face cache.
3. Record and pin the exact model repository revision in the experiment YAML.
4. Record the installed `transformers`, `peft`, `accelerate`, `bitsandbytes`,
   PyTorch, CUDA, and driver versions.
5. Reserve approximately 80-120 GB for the model cache, checkpoints, optimizer
   offload files, adapters, logs, and prediction exports.

## 3. Dataset and Evaluation Split

Reuse the repository's current binary mapping and study-level split.

| Split | T012a | T2b3 | Total |
|---|---:|---:|---:|
| Existing train | 128 | 133 | 261 |
| Existing validation | 80 | 70 | 150 |

For unbiased model selection:

1. Stratify the existing 261-study training set into an internal training and
   tuning split.
2. Preserve study/patient grouping and verify that no patient appears in more
   than one split.
3. Use the internal tuning set for early stopping, prompt selection, threshold
   selection, and hyperparameter selection.
4. Keep the existing 150-study validation set locked until final evaluation.

The previous VideoMAE result selected checkpoints on the 150-study validation
set. Therefore, comparisons against those reported results must state that the
selection protocols differ unless VideoMAE is rerun with the new internal
tuning split.

## 4. YAML-Driven Configuration

All MedGemma experiment configurations must inherit from one common data
configuration. Frame count must never be duplicated as a hard-coded constant
in the dataset, collator, prompt builder, model wrapper, or validation code.

Recommended common configuration:

```yaml
seed: 14207

data:
  label_mode: binary
  num_frames: 8
  image_size: 896
  interpolation: trilinear
  align_corners: true
  views: [AP, sagittal]

model:
  pretrained_name: google/medgemma-1.5-4b-it
  revision: null  # Replace with the pinned revision after download.

input:
  prompt_template: binary_tici_v1
  frame_order: chronological
```

`data.num_frames: 8` means eight frames **per DSA scan**. In the primary
dual-view experiment, each study has one AP scan and one sagittal scan, so the
model receives 16 ordered images per study. Changing `data.num_frames` later
must automatically update both scans, prompt frame markers, shape assertions,
and the expected visual-token estimate.

Separate YAML files should contain only experiment-specific overrides:

```text
medgemma_binary/config/common.yaml
medgemma_binary/config/zero_shot.yaml
medgemma_binary/config/frozen.yaml
medgemma_binary/config/lora.yaml
medgemma_binary/config/full_finetune.yaml
```

## 5. Dataloader Adaptation

### 5.1 Reuse the existing resampling behavior

Reuse `amticis_pipeline.transforms.ResizeView` rather than introducing uniform
frame indexing or a second temporal sampling implementation. For every active
DSA scan, the existing operation is:

```text
raw (H, W, T)
  -> permute to (1, 1, T, H, W)
  -> trilinear interpolation to
     (1, 1, data.num_frames, data.image_size, data.image_size)
  -> remove batch dimension
  -> (1, data.num_frames, data.image_size, data.image_size)
```

The initial value is `data.num_frames: 8`. The interpolation mode and
`align_corners` setting must come from YAML and default to the values already
used by the repository: `trilinear` and `true`.

Trilinear interpolation jointly resamples the temporal and spatial dimensions.
It must be applied independently to the AP and sagittal scans. Do not replace
this with nearest-frame selection, duplicated frames, or uniform integer
indices in the primary experiments.

### 5.2 Convert a resampled scan for MedGemma

After existing DSA resampling and augmentation:

1. Read the output tensor as `(1, num_frames, H, W)`.
2. Preserve temporal order when extracting frames.
3. Apply the selected DSA intensity conversion consistently across all frames
   in the scan.
4. Convert every grayscale frame to RGB by repeating the single channel three
   times.
5. Convert each frame to a processor-compatible PIL image or NumPy array.
6. Create ordered content blocks with the view name and frame number.
7. Pass the images and fixed prompt through MedGemma's `AutoProcessor`.
8. Return processor tensors, binary label, study name, view names, and frame
   count.

Do not use CT Hounsfield-unit windowing. It is part of Google's CT example and
is not appropriate for DSA.

### 5.3 Transform policy

Any random spatial transform must use identical sampled parameters for all
frames within a scan so it does not introduce artificial temporal motion.
Geometric consistency across AP and sagittal views is not required because
they are separate projections, but both views must use the same transform
policy.

Start with:

- Current DSA intensity normalization.
- Mild crop/affine augmentation only.
- No temporal reversal.
- No independent per-frame crop or rotation.
- No CT-specific channel construction.

Compare current `/255` normalization with per-scan percentile normalization
using only the internal tuning split.

### 5.4 Required dataloader tests

Tests should verify:

- `num_frames` is read from YAML.
- Changing it to 4 or 16 changes output and prompt lengths without code edits.
- Every scan has shape `(1, num_frames, image_size, image_size)` before RGB
  conversion.
- AP and sagittal scans are each resampled independently.
- Frame ordering is stable.
- Every frame becomes three-channel RGB.
- Labels exactly match the repository's existing binary mapping.
- The same study is not present across train, tuning, and final evaluation.
- A batch can complete one MedGemma forward pass.

## 6. Prompt and Model Interface

Use a fixed prompt describing the task and the ordering of the images. Prompt
wording must not contain information derived from a study label.

Conceptual input:

```text
Classify the final reperfusion outcome represented by these ordered DSA
frames. Class 0 represents mTICI T0, T1, or T2A. Class 1 represents mTICI T2B
or T3.

AP VIEW
FRAME 1 OF 8
<image>
...
FRAME 8 OF 8
<image>

SAGITTAL VIEW
FRAME 1 OF 8
<image>
...
FRAME 8 OF 8
<image>
```

The displayed frame total must be populated from `data.num_frames`, not typed
into the prompt template.

For the three trained classifiers:

1. Load `AutoModelForImageTextToText` with hidden-state output enabled.
2. Obtain the final hidden representation at the last non-padding prompt
   token.
3. Apply `LayerNorm -> Dropout -> Linear(hidden_size, 1)`.
4. Train using binary cross-entropy with logits.
5. Convert the output to the two-class pseudo-logit form already consumed by
   `ClassificationMetricAccumulator`.

The zero-shot experiment should compare the conditional likelihood of fixed
answers `0` and `1`, avoiding unconstrained free-form generation. It is a
reference baseline and does not share the randomly initialized classification
head used by the trained experiments.

## 7. Core Four-GPU Experiment

Run one independent strategy on each RTX 3090. All strategies must receive the
same resampled scans and final evaluation studies.

| GPU | Experiment | Trainable components |
|---:|---|---|
| 0 | E0: zero-shot | None |
| 1 | E1: frozen linear probe | Classification head only |
| 2 | E2: LoRA/QLoRA | Adapters and classification head |
| 3 | E3: full fine-tuning | Vision encoder, projector, language model, and head |

### 7.1 E0: Zero-shot

Purpose: determine whether the pretrained instruction-tuned model already
contains useful DSA/reperfusion discrimination.

Initial parameters:

- Deterministic inference.
- Score only fixed class answers.
- No explanatory generation.
- Evaluate two predefined prompt variants on the internal tuning split.
- Record invalid or tied answer scores.

### 7.2 E1: Frozen linear probe

Purpose: measure how separable the two classes are in fixed MedGemma features.

Initial parameters:

```yaml
training:
  max_epochs: 50
  early_stopping_patience: 10
  precision: bf16-mixed
  cache_frozen_features: true

optimizer:
  name: AdamW
  head_lr: 0.001
  weight_decay: 0.01

model:
  head_dropout: 0.2
```

Cache deterministic features once, then train the small classification head
with an effective batch size of 32-64.

### 7.3 E2: LoRA/QLoRA

Use QLoRA initially because the base model and multimodal activations must fit
on one 24 GB GPU.

```yaml
quantization:
  enabled: true
  bits: 4
  quant_type: nf4
  double_quant: true
  compute_dtype: bfloat16

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]

optimizer:
  adapter_lr: 0.0002
  head_lr: 0.001
  weight_decay: 0.01

training:
  micro_batch_size: 1
  gradient_accumulation_steps: 8
  max_epochs: 25
  warmup_ratio: 0.10
  scheduler: cosine
  gradient_clip_val: 1.0
  gradient_checkpointing: true
```

Start with the vision encoder frozen. Test language-only adapters versus
language adapters plus the multimodal projector after the core experiment.

### 7.4 E3: Full fine-tuning

True full fine-tuning is unlikely to fit natively on one RTX 3090 using normal
AdamW states. Plan for BF16 parameters, activation checkpointing, and CPU
optimizer offload.

```yaml
optimizer:
  name: AdamW8bit
  language_lr: 0.00001
  vision_lr: 0.000002
  projector_lr: 0.00001
  head_lr: 0.0001
  weight_decay: 0.01

training:
  micro_batch_size: 1
  gradient_accumulation_steps: 8
  max_epochs: 20
  early_stopping_patience: 5
  warmup_ratio: 0.10
  scheduler: cosine
  gradient_clip_val: 1.0
  precision: bf16-mixed
  gradient_checkpointing: true

deepspeed:
  zero_stage: 2
  offload_optimizer: cpu
```

If the full run cannot fit after optimizer offload and checkpointing, report
the failure and measured memory requirement. Do not quietly substitute QLoRA
and call it full fine-tuning. A later multi-GPU full-fine-tuning run would be a
separate experiment because it conflicts with assigning one GPU to each of
the four concurrent strategies.

## 8. Hyperparameter and Input Ablations

Run these only after completing the four core experiments.

### Input ablations

Change only YAML values:

| Parameter | Values |
|---|---|
| Frames per scan | 4, 8, 16 |
| Views | AP, sagittal, AP + sagittal |
| Intensity normalization | Existing `/255`, per-scan percentile |
| Prompt | Fixed variants v1 and v2 |

All frame-count variants must continue to use the same trilinear interpolation
implementation.

### LoRA ablations

| Parameter | Values |
|---|---|
| Rank | 8, 16, 32 |
| Alpha | 16, 32, 64 |
| Adapter LR | 1e-4, 2e-4, 4e-4 |
| Targets | Attention only, attention + projector, selected vision layers |

### Training ablations

- Effective batch size 4 versus 8.
- Classification-head dropout 0.1 versus 0.2 versus 0.5.
- Weighted sampler versus ordinary shuffled sampling.
- Fixed threshold 0.5 versus an internal-tuning-set threshold.
- Freeze the vision encoder for the first few epochs before unfreezing it.

## 9. Metrics and Evaluation Outputs

Reuse `amticis_training.classification_metrics.ClassificationMetricAccumulator`
and preserve its current naming conventions.

Primary metric:

- Macro F1 on the locked final evaluation set.

Existing metrics:

- Accuracy.
- Macro AUROC and AUPRC.
- Macro precision, recall, F1, and specificity.
- Per-class AUROC, AUPRC, precision, recall, F1, and specificity.
- TP, TN, FP, and FN.
- Confusion matrix.

Additional experiment diagnostics:

- BCE loss and train/tuning loss gap.
- Brier score and expected calibration error.
- Metrics at threshold 0.5 and at the internal-tuning-set threshold.
- Trainable and total parameter counts.
- Peak CPU and GPU memory.
- Samples per second and wall-clock time.
- Invalid/tied answer rate for zero-shot evaluation.

Export one row per study:

```text
name,true_label,probability_t2b3,predicted_label,threshold,experiment,seed
```

Calculate stratified bootstrap confidence intervals and paired bootstrap
differences against saved VideoMAE predictions. Do not interpret small metric
differences from point estimates alone on a 150-study evaluation set.

## 10. Execution Order

1. Verify gated model access and pin the model revision.
2. Add the common and experiment-specific YAML configurations.
3. Implement the MedGemma dataloader adapter around the existing
   `ResizeView` trilinear interpolation.
4. Add shape, label, ordering, configurability, and split-integrity tests.
5. Run one-study processor and forward-pass checks.
6. Benchmark inference and one backward pass at `num_frames: 8`.
7. Confirm that frozen, QLoRA, and full strategies fit their assigned GPUs.
8. Run E0-E3 concurrently using seed 14207.
9. Select checkpoints and thresholds only using the internal tuning split.
10. Evaluate once on the locked 150-study set and export predictions.
11. Repeat trainable experiments with at least two more seeds.
12. Perform frame-count, view, normalization, and LoRA ablations.
13. Compare against VideoMAE with confidence intervals and document the
    different model-selection protocols.

## 11. Runtime Estimate

The workstation has four RTX 3090 GPUs with 24 GB each. Estimates assume two
DSA scans per study, eight trilinearly interpolated frames per scan, 261
training studies, micro-batch size 1, and one experiment per GPU.

| Work | Estimated time |
|---|---:|
| Model download | 5-30 minutes |
| Dataloader and processor validation | 1-2 hours |
| E0 zero-shot evaluation | 0.5-1.5 hours |
| E1 frozen feature extraction and head training | 1-3 hours |
| E2 QLoRA, 20-30 epochs | 18-40 hours |
| E3 full fine-tuning with CPU offload | 30-70 hours |
| Final evaluation and exports | 1-2 hours |

Because E0-E3 run concurrently, one complete seed is expected to take roughly
32-74 hours, governed by full fine-tuning. Three seeds will require multiple
waves and approximately 4-9 days.

These ranges must be replaced after a measured one-epoch benchmark:

```text
estimated_hours =
    seconds_per_training_sample
    * number_of_training_studies
    * planned_epochs
    / 3600
```

Changing `data.num_frames` affects image-encoder work, multimodal context
length, VRAM use, and runtime. Every run must therefore log the resolved frame
count alongside its throughput and peak-memory measurements.

## 12. Completion Criteria

The experiment is complete when:

- All strategies use the same YAML-driven trilinear frame resampling.
- No MedGemma component contains a hard-coded expected frame count.
- The four core strategies have predictions for the same locked studies.
- At least three seeds are available for each trainable strategy.
- Existing metrics, calibration metrics, confidence intervals, memory, and
  runtime are reported.
- Checkpoints, resolved configurations, model revision, predictions, and
  environment versions are preserved.
- Limitations clearly state that MedGemma is not clinically validated for this
  task and that its 3D radiology training is not equivalent to DSA temporal
  angiography training.
