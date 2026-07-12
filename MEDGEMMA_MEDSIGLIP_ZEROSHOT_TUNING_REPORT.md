# MedGemma and MedSigLIP Zero-Shot Binary TICI Tuning Report

## 1. Scope and Status

This report documents the corrected zero-shot tuning experiments for binary
mTICI classification on the AmTICIS DSA dataset. It covers:

- `google/medgemma-1.5-4b-it`
- `google/medsiglip-448`
- two predefined prompt variants
- percentile and min-max DSA intensity scaling
- the internal 53-study tuning split

The locked 150-study final split has **not** been evaluated. Therefore, the
results below are model-selection results and must not be reported as final
generalization performance.

Binary labels are:

| Class | Label | Original grades |
|---|---:|---|
| T012a | 0 | T0, T1, T2A |
| T2b3 | 1 | T2B, T3 |

## 2. Files

### Implementation and configuration

- [MedGemma runner](medgemma_binary/zero_shot.py)
- [MedSigLIP runner](medgemma_binary/medsiglip_zero_shot.py)
- [Shared evaluation utilities](medgemma_binary/common.py)
- [MedGemma configuration](medgemma_binary/config_zero_shot.yaml)
- [MedSigLIP configuration](medgemma_binary/config_medsiglip_zero_shot.yaml)
- [Isolated environment requirements](medgemma_binary/requirements-medgemma.txt)
- [Regression and processor tests](tests/test_medgemma_zero_shot.py)
- [Original failed-run report](MEDGEMMA_ZEROSHOT_BINARY_TICI_REPORT.md)

### Generated artifacts

Generated artifacts are kept under the ignored `output_runs_lightning/`
directory because per-study CSV files contain study identifiers.

```text
output_runs_lightning/
├── medgemma_internal_split.json
├── medgemma_zero_shot_binary_tici_corrected/
│   ├── resolved_config.yaml
│   ├── environment.yaml
│   ├── selection.yaml
│   ├── tuning__<prompt>__<scaling>_scores.csv
│   ├── tuning__<prompt>__<scaling>_metrics.yaml
│   └── tuning__<prompt>__<scaling>_confusion_matrix.png
└── medsiglip_zero_shot_binary_tici_control/
    └── corresponding configuration, selection, score, metric, and plot files
```

## 3. Dataset and Split

The experiment reused `train_val_split.json` through the existing
`AmTICISDataModule`. Patient IDs are disjoint across all partitions.

| Partition | Class 0 | Class 1 | Total | Use |
|---|---:|---:|---:|---|
| Development | 102 | 106 | 208 | Reserved for later adaptation work |
| Tuning | 26 | 27 | 53 | Prompt and scaling selection |
| Locked final | 80 | 70 | 150 | Not evaluated |

The internal split was generated with stratification and seed `14207`. The
same tuning studies were used for both models.

## 4. Data Loader and Preprocessing

For each study, the loader independently reads AP and sagittal NIfTI scans as
raw `(H, W, T)` arrays and applies the repository's `ResizeView` operation:

```text
(H, W, T)
  -> permute to (1, 1, T, H, W)
  -> one trilinear interpolation over time and space
  -> (1, 8, model_image_size, model_image_size)
```

Common loader parameters:

| Parameter | Value |
|---|---|
| Active views | AP, sagittal |
| Frames per view | 8 |
| Total images per study | 16 |
| Frame order | Chronological |
| Interpolation | Trilinear |
| `align_corners` | `true` |
| Evaluation transforms | None |
| Batch size | 1 |
| Workers | 4 |
| Pin memory | `false` |
| Weighted sampler | Disabled |

After interpolation, each scan is scaled consistently across all its frames
using one of the following predefined policies:

- `percentile_1_99`: clip to the scan's 1st and 99th percentiles, then map to
  `[0, 1]`.
- `minmax`: map the scan minimum and maximum to `[0, 1]`.

Frames are converted from grayscale to RGB by repeating the channel. No
random augmentation or CT-specific windowing is applied.

Model-specific processor inputs:

| Model | Spatial input | Processor tensor |
|---|---:|---|
| MedGemma | 896 x 896 | `(16, 3, 896, 896)` |
| MedSigLIP | 448 x 448 | `(16, 3, 448, 448)` |

## 5. Model and Scoring Configuration

### Runtime environment

| Parameter | Value |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.57.1 |
| Accelerate | 1.12.0 |
| CUDA runtime | 12.4 |
| Hardware | NVIDIA RTX 3090, 24 GiB |
| Precision | Native BF16 |
| Quantization | None |
| Seed | 14207 |

### MedGemma

| Parameter | Value |
|---|---|
| Checkpoint | `google/medgemma-1.5-4b-it` |
| Revision | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` |
| Attention implementation | SDPA |
| Device placement | Accelerate `device_map: auto` |
| Maximum GPU allocation | 20 GiB per configured GPU |
| CPU allocation limit | 64 GiB |
| Fast image processor | Disabled |

MedGemma receives view and frame markers followed by a binary classification
question. The fixed answers `0` and `1` are verified to be distinct single
tokens. One multimodal forward pass produces the final hidden state, and only
the two answer-token rows of the language-model head are projected with FP32
accumulation. A two-class softmax converts these scores into the exported
class-1 pseudo-probability. This replaced the invalid earlier implementation,
which silently converted non-finite scores to `-1e9`.

### MedSigLIP

| Parameter | Value |
|---|---|
| Checkpoint | `google/medsiglip-448` |
| Revision | `9cea28a1a1195f665105faa6e8544c112fd960a4` |
| Device placement | Accelerate `device_map: auto` |
| Fast image processor | Disabled |

MedSigLIP compares each frame against two class descriptions. The 16 per-frame
class logits are averaged, then normalized with a two-class softmax. AP-only
and sagittal-only probabilities are also exported. This aggregation is
intentionally order-invariant and is a medical image-encoder control rather
than a temporal sequence model.

### Prompt variants

- `clinical_v1`: explicitly describes unsuccessful mTICI 0/1/2A and successful
  mTICI 2B/3.
- `concise_v2`: uses a shorter DSA/reperfusion formulation.

The exact MedGemma prompts and MedSigLIP text labels are preserved in the two
configuration files.

## 6. Evaluation Parameters

| Parameter | Value |
|---|---|
| Decision threshold | Class 1 probability >= 0.5 |
| Selection metric | Macro F1, then AUROC as tie-breaker |
| Bootstrap | 10,000 stratified replicates |
| Confidence interval | 95% percentile interval |
| Maximum tie-rate diagnostic | 1% |
| Invalid-score handling | Abort; no numeric replacement |

Reported metrics include accuracy, balanced accuracy, macro F1, AUROC, AUPRC,
per-class precision/recall/F1/specificity, exact tie rate, runtime, and peak
allocated memory.

Because the tuning split contains 26 class-0 and 27 class-1 studies, the
constant class-1 baseline is:

| Accuracy | Balanced accuracy | Macro F1 |
|---:|---:|---:|
| 0.509 | 0.500 | 0.338 |

## 7. Results

### MedGemma variants

| Prompt | Scaling | Accuracy | Balanced acc. | Macro F1 | AUROC | AUPRC | Tie rate |
|---|---|---:|---:|---:|---:|---:|---:|
| clinical_v1 | min-max | 0.509 | 0.500 | 0.338 | 0.454 | 0.493 | 0.000 |
| clinical_v1 | percentile | 0.528 | 0.519 | 0.379 | 0.530 | 0.533 | 0.000 |
| concise_v2 | min-max | 0.453 | 0.457 | 0.433 | 0.417 | 0.461 | 0.000 |
| **concise_v2** | **percentile** | **0.453** | **0.456** | **0.440** | **0.403** | **0.451** | **0.000** |

The selection rule chose `concise_v2 + percentile_1_99` because it had the
highest macro F1. Its bootstrap intervals were:

| Metric | Point estimate | 95% CI |
|---|---:|---:|
| Macro F1 | 0.440 | 0.309-0.573 |
| Balanced accuracy | 0.456 | 0.324-0.588 |
| AUROC | 0.403 | 0.256-0.561 |

This selected variant does not demonstrate reliable discrimination: accuracy
and balanced accuracy are below their baselines, AUROC is below 0.5, and all
confidence intervals include the corresponding baseline or chance value. The
clinical prompt with percentile scaling had the highest AUROC, but predicted
class 1 for 52 of 53 studies.

Total runtime for the four MedGemma variants was approximately 14.9 minutes.
The recorded peak allocated GPU memory was approximately 4.0 GB; peak process
resident memory was approximately 6.3 GB.

### MedSigLIP variants

| Prompt | Scaling | Accuracy | Balanced acc. | Macro F1 | AUROC | AUPRC | Tie rate |
|---|---|---:|---:|---:|---:|---:|---:|
| **clinical_v1** | **min-max** | **0.509** | **0.518** | **0.396** | **0.516** | **0.537** | **0.019** |
| clinical_v1 | percentile | 0.472 | 0.481 | 0.321 | 0.528 | 0.502 | 0.000 |
| concise_v2 | min-max | 0.509 | 0.500 | 0.338 | 0.412 | 0.486 | 0.000 |
| concise_v2 | percentile | 0.509 | 0.500 | 0.338 | 0.462 | 0.502 | 0.000 |

The selection rule chose `clinical_v1 + minmax`. Its bootstrap intervals were:

| Metric | Point estimate | 95% CI |
|---|---:|---:|
| Macro F1 | 0.396 | 0.312-0.494 |
| Balanced accuracy | 0.518 | 0.461-0.574 |
| AUROC | 0.516 | 0.355-0.677 |

The selected variant predicted class 0 for 50 of 53 studies and recalled only
2 of 27 successful-reperfusion studies. Its 1.9% exact-tie rate exceeded the
configured 1% diagnostic threshold. The highest-AUROC variant still recalled
zero class-1 studies at the fixed threshold.

MedSigLIP probabilities were tightly concentrated around 0.5. For the selected
variant, class-1 probabilities ranged from 0.4849 to 0.5005 with standard
deviation 0.0035.

Total runtime for the four MedSigLIP variants was approximately 5.7 minutes.
The recorded post-reset peak GPU allocation was approximately 1.0 GB; peak
process resident memory was approximately 4.7 GB.

## 8. Confusion Matrices

Rows are actual classes and columns are predicted classes.

### Selected MedGemma configuration

`concise_v2 + percentile_1_99`

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 16 | 10 |
| Class 1 | 19 | 8 |

### Selected MedSigLIP configuration

`clinical_v1 + minmax`

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 25 | 1 |
| Class 1 | 25 | 2 |

### All configurations

| Model and variant | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| MedGemma clinical/min-max | 0 | 26 | 0 | 27 |
| MedGemma clinical/percentile | 1 | 25 | 0 | 27 |
| MedGemma concise/min-max | 17 | 9 | 20 | 7 |
| MedGemma concise/percentile | 16 | 10 | 19 | 8 |
| MedSigLIP clinical/min-max | 25 | 1 | 25 | 2 |
| MedSigLIP clinical/percentile | 25 | 1 | 27 | 0 |
| MedSigLIP concise/min-max | 0 | 26 | 0 | 27 |
| MedSigLIP concise/percentile | 0 | 26 | 0 | 27 |

## 9. Conclusion

The corrected pipelines are numerically valid: all studies produced finite
scores, the original all-`-1e9` failure is removed, model revisions and inputs
are pinned, and the full intended 16-image input fits available hardware.

However, neither zero-shot model demonstrates a credible classification
improvement on the tuning split:

- MedGemma's selected macro-F1 improvement is accompanied by below-chance
  balanced accuracy and AUROC.
- MedSigLIP's selected result is nearly chance-level and strongly collapses to
  class 0.
- Confidence intervals for both selected variants include chance or the
  majority baseline.
- Prompt changes materially alter the predicted class distribution, indicating
  prompt/verbalizer bias rather than stable DSA discrimination.

If the purpose of final evaluation is to advance only models showing tuning
improvement, the `phase final` runs should not be performed. If a complete,
unbiased negative zero-shot benchmark is required for publication, the locked
final split may be evaluated once with the already saved selections, explicitly
reported as a confirmatory negative endpoint.

The practical next experiment should be a frozen MedSigLIP/MedGemma feature
probe or a carefully regularized adaptation experiment rather than additional
zero-shot prompt selection.

## 10. Limitations

- Only 53 studies were available for prompt and scaling selection, producing
  wide confidence intervals and substantial selection uncertainty.
- Four variants were compared on the same tuning set; the selected point
  estimate is optimistically biased even without further prompt iteration.
- Neither model is specifically validated or pretrained for temporal DSA
  reperfusion grading.
- MedSigLIP averages independent frame logits and cannot model temporal order.
- MedGemma pseudo-probabilities are normalized only across two fixed answer
  tokens and should not be interpreted as calibrated clinical probabilities.
- MedSigLIP probabilities are highly compressed around 0.5, making the fixed
  threshold sensitive to small numerical changes.
- Trilinear temporal interpolation standardizes sequence length but may blur
  short contrast-flow events.
- AP and sagittal views are treated as two ordered image groups without an
  explicit learned cross-view or temporal alignment mechanism.
- The final 150-study split has deliberately not been examined, so no final
  performance or external-validity claim can be made.
- These models and results are not suitable for clinical use.
