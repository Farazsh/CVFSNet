# MedGemma Zero-Shot Binary TICI Failed-Run Report

> **INVALID EXPERIMENT:** Every exported class score in this run is `-1e9`
> because non-finite model outputs were replaced by a numeric fallback. The
> resulting `0.5` probabilities and class-0 predictions contain no model
> signal. None of the performance values below should be cited or compared
> with another model. This file is retained only as a failure record.

## Corrected Pipeline Status

The replacement pipeline was validated on 2026-07-12 with a balanced two-study
smoke set. This is an implementation check, not a performance estimate.

- checkpoint revision: `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`
- environment: PyTorch `2.6.0+cu124`, Transformers `4.57.1`, native BF16
- input: 8 chronological frames from each AP and sagittal view at `896x896`
- finite-score rate: 2/2; exact ties: 0/2
- peak allocated GPU memory: approximately 4.0 GB
- real, reversed, blank, and text-only controls produced different score margins
- corrected smoke artifacts: `output_runs_lightning/medgemma_zero_shot_binary_tici_corrected/`

After the initial implementation, BF16 full-vocabulary projection produced
exact class-score ties in 5.7% of one tuning variant. The corrected scorer now
projects only the two fixed answer tokens with FP32 accumulation. A balanced
10-study smoke run then produced finite scores for all studies with zero exact
ties. Tie-rate threshold violations are retained as diagnostics rather than
discarding completed predictions.

The revision-pinned MedSigLIP control was subsequently validated with the same
balanced two-study smoke set:

- checkpoint revision: `9cea28a1a1195f665105faa6e8544c112fd960a4`
- input: 8 frames from each AP and sagittal view at native `448x448`
- finite-score rate: 2/2; exact ties: 0/2
- peak allocated GPU memory: approximately 2.2 GB
- real and blank-image controls produced different score margins
- corrected smoke artifacts: `output_runs_lightning/medsiglip_zero_shot_binary_tici_control/`

MedSigLIP averages independent per-frame logits, so reversing frame order is
expected to leave its control score unchanged. It is not a temporal model.

## Overview

This report documents the zero-shot binary TICI classification experiment run with `google/medgemma-1.5-4b-it` on the AmTICIS validation split.

The experiment followed the repository's existing evaluation pipeline and produced:

- local per-sample scores in CSV format
- aggregated metrics in YAML format
- a confusion matrix image
- a W&B run with the same metrics and confusion matrix

The evaluation was completed successfully, but it required a reduced fallback configuration to fit the available GPU memory.

## Files

### Experiment code

- [medgemma_binary/zero_shot.py](/home/fshaikh/Projects/CVFSNet/medgemma_binary/zero_shot.py)
- [medgemma_binary/config_zero_shot.yaml](/home/fshaikh/Projects/CVFSNet/medgemma_binary/config_zero_shot.yaml)

### Output artifacts

- [output_runs_lightning/medgemma_zero_shot_binary_tici/val_zero_shot_scores.csv](/home/fshaikh/Projects/CVFSNet/output_runs_lightning/medgemma_zero_shot_binary_tici/val_zero_shot_scores.csv)
- [output_runs_lightning/medgemma_zero_shot_binary_tici/val_zero_shot_metrics.yaml](/home/fshaikh/Projects/CVFSNet/output_runs_lightning/medgemma_zero_shot_binary_tici/val_zero_shot_metrics.yaml)
- [output_runs_lightning/medgemma_zero_shot_binary_tici/val_confusion_matrix.png](/home/fshaikh/Projects/CVFSNet/output_runs_lightning/medgemma_zero_shot_binary_tici/val_confusion_matrix.png)
- [output_runs_lightning/medgemma_zero_shot_binary_tici/resolved_zero_shot_config.yaml](/home/fshaikh/Projects/CVFSNet/output_runs_lightning/medgemma_zero_shot_binary_tici/resolved_zero_shot_config.yaml)

## Data Loader Details

The experiment reused the repository's existing `AmTICISDataModule`.

Key pipeline behavior:

- `AmTICISDataModule` builds train/val/test datasets from the split JSON and the configured view set.
- The resize step is `ResizeView`, which performs a single 3D trilinear interpolation from raw `(H, W, T)` scans into a fixed `(1, num_frames, image_size, image_size)` tensor.
- Frame count and image size remain YAML-driven through `data.num_frames` and `data.image_size`.
- The active views for this experiment were `AP` and `sagittal`.
- Batch size was `1` for validation, with `val_num_workers: 4`.

Relevant implementation:

- `amticis_pipeline/transforms.py`
- `amticis_pipeline/dataset_loader.py`

### Practical note

Although the original plan targeted a higher-resolution / higher-frame configuration, the successful run used a fallback of:

- `num_frames: 1`
- `image_size: 128`

This was necessary to keep the MedGemma forward pass within the available GPU memory.

## Configuration Details

### Model

- pretrained model: `google/medgemma-1.5-4b-it`
- precision: `float16`
- quantization: `4bit`
- 4-bit settings:
  - `bnb_4bit_compute_dtype: float16`
  - `bnb_4bit_quant_type: nf4`
  - `bnb_4bit_use_double_quant: true`
- device placement:
  - `model.vision_tower: 0`
  - `model.multi_modal_projector: 0`
  - `model.language_model: 3`
  - `lm_head: 3`
- offload folder: `.offload_medgemma`

### Evaluation

- split: `val`
- answer labels: `["0", "1"]`
- scoring mode: fixed-answer likelihood comparison
- intensity scaling:
  - percentile mode
  - lower: `1.0`
  - upper: `99.0`

### Logging

- W&B enabled: `true`
- project: `AmTICIS`
- tags:
  - `medgemma`
  - `zero-shot`
  - `binary`
  - `tici`
  - `dsa`

## Parameters

Important parameters used in the final successful run:

- `seed: 14207`
- `num_frames: 1`
- `image_size: 128`
- `interpolation: trilinear`
- `align_corners: true`
- `batch_size: 1`
- `val_batch_size: 1`
- `num_workers: 4`
- `val_num_workers: 4`
- `quantization: 4bit`
- `device_map` split across GPU 0 and GPU 3

## Result

Validation set summary:

- samples: `150`
- accuracy: `0.5333333333333333`
- macro F1: `0.34782608695652173`
- AUROC: `0.5`

Metric breakdown from `val_zero_shot_metrics.yaml`:

- `fuse/accuracy: 0.5333333333333333`
- `fuse/class_0/precision: 0.5333333333333333`
- `fuse/class_0/recall: 1.0`
- `fuse/class_0/f1: 0.6956521739130436`
- `fuse/class_1/precision: 0.0`
- `fuse/class_1/recall: 0.0`
- `fuse/class_1/f1: 0.0`
- `fuse/f1_macro: 0.3478260869565218`
- `fuse/auroc_macro: 0.5`

## Confusion Matrix

The model predicted class `0` for every validation sample.

Raw confusion matrix:

| Actual \ Predicted | Class 0 | Class 1 |
| --- | ---: | ---: |
| Class 0 | 80 | 0 |
| Class 1 | 70 | 0 |

Normalized row-wise matrix:

| Actual \ Predicted | Class 0 | Class 1 |
| --- | ---: | ---: |
| Class 0 | 1.00 | 0.00 |
| Class 1 | 1.00 | 0.00 |

The rendered plot is stored at:

- [val_confusion_matrix.png](/home/fshaikh/Projects/CVFSNet/output_runs_lightning/medgemma_zero_shot_binary_tici/val_confusion_matrix.png)

## Conclusion

The MedGemma zero-shot pipeline is now wired end-to-end for binary TICI classification, with:

- reusable YAML configuration
- existing AmTICIS data preprocessing
- local CSV and metrics export
- confusion matrix generation
- W&B tracking

The final run completed successfully and produced a reproducible validation report. The observed performance is modest, and the model collapsed to predicting the majority class under the fallback inference configuration.

## Limitations

- The originally intended higher-resolution / multi-frame configuration did not fit the available GPU memory in this environment.
- The final run used a very aggressive fallback (`1` frame, `128 px`, 4-bit quantization) to ensure completion.
- Because of the fallback configuration, the measured performance should be treated as a baseline rather than the final intended MedGemma setup.
- The model produced a degenerate prediction pattern, which limits the interpretability of the current zero-shot result.
- The run depends on Hugging Face authentication for the gated MedGemma repository.

## Notes for Follow-Up

If the goal is to move closer to the planned configuration, the next steps would be:

1. Increase `num_frames` gradually.
2. Increase `image_size` gradually.
3. Re-test with tighter memory budgets or more aggressive offload.
4. Compare against the planned fine-tuning variants once the zero-shot baseline is stable.
