# CVFSNet Project Overview

CVFSNet is a medical-video classification research project for automatically
grading reperfusion after mechanical thrombectomy from Digital Subtraction
Angiography (DSA). It uses the AmTICIS dataset and supports coronal/AP,
sagittal, and dual-view inputs.

The repository currently contains four related systems:

1. The original CVFSNet implementation and training loop.
2. A configuration-driven PyTorch Lightning implementation of CVFSNet.
3. VideoMAE experiments for ordinal and binary mTICI classification.
4. MedGemma and MedSigLIP zero-shot binary mTICI experiments.

The original project documentation is preserved in
[`readme_old.md`](readme_old.md).

## Clinical Labels

Labels are derived from scan filenames and can be represented at three
granularities:

| Mode | Classes |
| --- | --- |
| `full` | T0, T1, T2a, T2b, T3 |
| `fuse01` | T0/T1, T2a, T2b, T3 |
| `binary` | T0/T1/T2a vs. T2b/T3 |

The mappings are implemented in
[`amticis_pipeline/utils.py`](amticis_pipeline/utils.py). The binary boundary
represents unsuccessful versus successful reperfusion.

## Repository Structure

| Path | Responsibility |
| --- | --- |
| [`Src/CVFSNet.py`](Src/CVFSNet.py) | Original CVFSNet and CVFM architecture |
| [`Data/`](Data/) | Legacy dataset and augmentation code |
| [`Loss/`](Loss/) | Legacy imbalance and multi-output losses |
| [`main.py`](main.py) | Legacy training entry point |
| [`infer.py`](infer.py) | Legacy checkpoint evaluation |
| [`amticis_pipeline/`](amticis_pipeline/) | Structured dataset, transforms, configuration, and Lightning DataModule |
| [`amticis_training/`](amticis_training/) | Modern Lightning CVFSNet training stack |
| [`mae_ordinal/`](mae_ordinal/) | VideoMAE ordinal and binary experiments |
| [`medgemma_binary/`](medgemma_binary/) | MedGemma and MedSigLIP zero-shot binary experiments |
| [`plotting_functions/`](plotting_functions/) | Dataset and augmentation visualization tools |
| [`mae_ordinal/docs/ORDINAL_RESULTS_REPORT.md`](mae_ordinal/docs/ORDINAL_RESULTS_REPORT.md) | Ordinal experiment results |
| [`mae_ordinal/docs/BINARY_RESULTS_REPORT.md`](mae_ordinal/docs/BINARY_RESULTS_REPORT.md) | Binary experiment results |
| [`mae_ordinal/docs/COMPREHENSIVE_EXPERIMENT_REPORT.md`](mae_ordinal/docs/COMPREHENSIVE_EXPERIMENT_REPORT.md) | Comprehensive VideoMAE experiment report |
| [`medgemma_binary/docs/`](medgemma_binary/docs/) | MedGemma plans and reports |

## Dataset Pipeline

The reusable pipeline is controlled by
[`amticis_pipeline/config.yaml`](amticis_pipeline/config.yaml).

```text
train_val_split.json
    -> resolve AP/sagittal NIfTI filename
    -> load float32 volume with SimpleITK
    -> convert to (1, T, H, W)
    -> trilinear temporal/spatial resampling
    -> training augmentation pipeline
    -> normalization
    -> dictionary containing views, label, and scan name
```

The dataset returns one tensor for every configured view, a label tensor, and
the source scan name. View filenames are resolved from the coronal filename by
replacing the `_C` marker with the configured view marker.

Default preprocessing includes:

- 8 frames at 256 x 256 for CVFSNet.
- 16 frames at 224 x 224 for VideoMAE.
- Morphology, flipping, motion, ghosting, spikes, blur, noise, gamma, rotation,
  and crop augmentation during training.
- Deterministic validation normalization.
- Inverse-frequency `WeightedRandomSampler` sampling during training.

The Lightning integration is implemented in
[`amticis_pipeline/dataset_loader.py`](amticis_pipeline/dataset_loader.py).

## Dataset Statistics

The configured split contains 411 studies with no patient-ID overlap between
training and validation:

| Split | T0 | T1 | T2a | T2b | T3 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 80 | 18 | 30 | 48 | 85 | 261 |
| Validation | 37 | 14 | 29 | 23 | 47 | 150 |

Binary distribution:

| Split | T0/T1/T2a | T2b/T3 |
| --- | ---: | ---: |
| Train | 128 | 133 |
| Validation | 80 | 70 |

There is no independent test cohort. The current data configuration maps the
logical `test` split to `val`, which is important when interpreting reported
results.

## CVFSNet Architecture

CVFSNet consists of:

- Two independent X3D-style spatiotemporal CNN branches.
- One branch for AP/coronal video and one for sagittal video.
- Optional auxiliary classification heads for deep supervision.
- A Cross View Fusion Module (CVFM).
- Separate coronal, sagittal, and fused predictions.

CVFM combines intermediate branch features using a learned angular or
pseudo-oblique feature, a Pythagorean feature magnitude, multi-scale attention,
and projected spatiotemporal pooling. The resulting representation is passed to
a fused classification head.

The modern CVFSNet training stack supports registry-based model and loss
construction, typed pipeline configuration, named multi-output metrics, CSV and
Weights & Biases logging, checkpointing, and distributed training. Its default
loss combines label smoothing and Seesaw cross-entropy across the fused,
per-view, and enabled auxiliary outputs.

## VideoMAE Architecture

The VideoMAE experiments use the Kinetics-pretrained Hugging Face
`VideoMAEModel`.

The input adapter performs:

```text
grayscale clip
    -> replicate one channel to RGB
    -> per-clip min-max normalization
    -> ImageNet mean/std normalization
    -> VideoMAE encoder
```

This adapter is essential because the pipeline's original normalized DSA
intensities have much lower variance than the pretrained VideoMAE model expects.

### Ordinal Model

The ordinal model uses mean-pooled VideoMAE tokens followed by LayerNorm,
dropout, and a linear head producing `K - 1` CORN conditional ordinal logits.
In addition to classification metrics, it records mean absolute error,
off-by-one accuracy, and quadratic weighted kappa.

The relevant implementation is in
[`mae_ordinal/model.py`](mae_ordinal/model.py) and
[`mae_ordinal/ordinal.py`](mae_ordinal/ordinal.py).

### Binary Models

Three binary configurations are provided:

- AP-only classification.
- Sagittal-only classification.
- Dual-view classification with a shared VideoMAE encoder and late
  concatenation of AP and sagittal features.

## Training Workflows

Dependencies are managed through `uv` in [`pyproject.toml`](pyproject.toml).
The environment uses Python 3.9-3.12, PyTorch 2.4.1, CUDA 12.1 wheels,
Lightning, Transformers, TorchIO, SimpleITK, and scikit-learn.

Install or synchronize the environment:

```bash
uv sync
```

### Modern CVFSNet

```bash
uv run python -m amticis_training.train
```

Configuration values can be overridden with dotted keys:

```bash
uv run python -m amticis_training.train \
  --set trainer.max_epochs=2 logging.wandb.enabled=false
```

The default configuration is
[`amticis_training/config.yaml`](amticis_training/config.yaml).

### VideoMAE Ordinal

Frozen backbone:

```bash
uv run python -m mae_ordinal.train \
  --config mae_ordinal/config_frozen.yaml
```

Full fine-tuning:

```bash
uv run python -m mae_ordinal.train \
  --config mae_ordinal/config_finetune.yaml
```

### VideoMAE Binary

```bash
uv run python -m mae_ordinal.train \
  --config mae_ordinal/config_binary_ap.yaml

uv run python -m mae_ordinal.train \
  --config mae_ordinal/config_binary_sagittal.yaml

uv run python -m mae_ordinal.train \
  --config mae_ordinal/config_binary_dual.yaml
```

Evaluate the binary checkpoints and regenerate plots with:

```bash
uv run python -m mae_ordinal.eval_binary
uv run python -m mae_ordinal.plot_confusion_binary
```

### Legacy CVFSNet

The original workflow remains available through `main.py`, YACS configuration,
and `infer.py`. See [`readme_old.md`](readme_old.md) for its command-line
options and output layout.

## Recorded Results

### Ordinal AP-Only Validation

| Model | Accuracy | Macro F1 | QWK |
| --- | ---: | ---: | ---: |
| Frozen VideoMAE | 0.447 | 0.425 | 0.346 |
| Fine-tuned VideoMAE | 0.573-0.580 | 0.497 | 0.647-0.648 |
| Coronal CVFSNet | 0.547 | 0.479 | 0.698 |

Fine-tuned VideoMAE improves accuracy and macro F1 over the coronal CVFSNet
baseline, but CVFSNet has better ordinal agreement. The fine-tuned VideoMAE
model strongly favors T3 and has relatively weak T2a/T2b recall.

### Binary Validation

| VideoMAE input | Accuracy | Macro F1 | AUROC |
| --- | ---: | ---: | ---: |
| AP | 0.827 | 0.826 | 0.925 |
| Sagittal | 0.780 | 0.779 | 0.845 |
| Dual | 0.853 | 0.853 | 0.895 |

The dual-view binary model is the strongest recorded classifier. These results
remain exploratory because checkpoint selection and reporting use the same
150-study validation cohort.

Detailed results and limitations are documented in
[`ORDINAL_RESULTS_REPORT.md`](mae_ordinal/docs/ORDINAL_RESULTS_REPORT.md),
[`BINARY_RESULTS_REPORT.md`](mae_ordinal/docs/BINARY_RESULTS_REPORT.md), and
[`COMPREHENSIVE_EXPERIMENT_REPORT.md`](mae_ordinal/docs/COMPREHENSIVE_EXPERIMENT_REPORT.md).

## Testing

Run the unit tests with:

```bash
uv run python -m unittest discover -s tests -v
```

The current tests cover training-configuration merging and classification
metrics. Dataset loading, transforms, model forward passes, losses, checkpoint
loading, and end-to-end training are not yet covered.

## Known Limitations and Risks

- There is no held-out test set; `test` currently aliases `val`.
- Experiments generally use one random seed and best-validation checkpoint
  selection.
- The repository retains both legacy and modern preprocessing and training
  implementations, which creates duplicated behavior and maintenance cost.
- The default Lightning CVFSNet configuration currently maps both `AP` and
  `sagittal` view definitions to `_S`. It must map `AP` from `_C` to `_C` before
  being used for a genuine dual-view run.
- Several training configurations contain machine-specific GPU device IDs.
- The small 261-study training cohort is challenging for an approximately
  86-million-parameter VideoMAE backbone and is prone to overfitting.
- CVFSNet and VideoMAE comparisons use related but not always identical
  objectives, particularly when four-class CVFSNet predictions are collapsed
  into binary predictions after training.

## License and Citation

The code and AmTICIS dataset are released under the Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International license. See
[`LICENSE`](LICENSE) for the full terms.

If using this project, cite:

> Xu, W., Tan, T., Yang, H., Liu, W., Chen, Y., Zhang, L., et al. (2025).
> CVFSNet: A Cross View Fusion Scoring Network for end-to-end mTICI scoring.
> Medical Image Analysis, 102, 103508.
