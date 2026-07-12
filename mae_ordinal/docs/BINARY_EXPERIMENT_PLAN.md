# VideoMAE Binary TICI Classification Experiment Plan

**Branch:** `mae-binary-mix`  
**Status:** Implemented; training and evaluation complete.

---

## 1. Problem definition

| Class | Raw mTICI tokens | Integer label | Meaning |
|-------|------------------|---------------|---------|
| **T012a** (class 0) | T0, T1, T2A | 0 | Incomplete / partial reperfusion |
| **T2b3** (class 1) | T2B, T3 | 1 | Near-complete / complete reperfusion |

This matches the repo's existing **`label_mode: binary`** in
[`amticis_pipeline/utils.py`](../../amticis_pipeline/utils.py) (`_BINARY_LABELS`).

**Class distribution (same split as prior experiments):**

| split | T012a (0) | T2b3 (1) | total |
|-------|----------:|---------:|------:|
| train | 128 | 133 | 261 |
| val   | 80  | 70  | 150 |

(`test = val`, seed 14207.)

---

## 2. Reuse from prior VideoMAE work

Build on [`mae_ordinal/`](../) without modifying `amticis_pipeline/`,
`amticis_training/`, `Src/`, `Data/`, `Loss/`, or `Lib/`.

| Component | Reuse | Binary change |
|-----------|-------|---------------|
| Data pipeline | `AmTICISDataModule` | View presets + `label_mode: binary` |
| Preprocessing | Trilinear 16×224², DSA aug, `TioZNormalization(div255)` | Same |
| VideoMAE adapter | Min-max → ImageNet mean/std | Same |
| Sampling | `WeightedRandomSampler` | Same |
| Metrics | `ClassificationMetricAccumulator`, `val/fuse/*` | 2-class BCE |
| Hyperparameters | Ordinal fine-tune recipe | Full fine-tune only |

**Training recipe:** AdamW `base_lr=2e-4`, LLRD 0.75, `wd=0.05`, warmup 5 + cosine, batch 8 (single-view) / 4 (dual).

---

## 3. Architecture

### Single-view (`VideoMAEBinary`)

VideoMAE-Base → mean-pool → LayerNorm → Dropout → Linear(1) → BCE loss.

### Dual-view (`VideoMAEBinaryDual`)

Shared backbone encodes AP and sagittal separately; pooled features concatenated (1536-d) → binary head.

---

## 4. Experiments

| Config | Run name | View | GPU |
|--------|----------|------|-----|
| `config_binary_ap.yaml` | `mae_binary_t012a_ap` | AP | 0 |
| `config_binary_sagittal.yaml` | `mae_binary_t012a_sag` | sagittal | 1 |
| `config_binary_dual.yaml` | `mae_binary_t012a_dual` | AP + sagittal | 2 |

Launch:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m mae_ordinal.train --config mae_ordinal/config_binary_ap.yaml
CUDA_VISIBLE_DEVICES=1 uv run python -m mae_ordinal.train --config mae_ordinal/config_binary_sagittal.yaml
CUDA_VISIBLE_DEVICES=2 uv run python -m mae_ordinal.train --config mae_ordinal/config_binary_dual.yaml
```

---

## 5. Post-training evaluation

**Sigmoid CSV** (per run, val set):
```bash
uv run python -m mae_ordinal.eval_binary
```
→ `output_runs_lightning/<run>/val_sigmoid_scores.csv` with columns `name`, `true_label`, `sigmoid_score`, `pred_label`.

**Confusion matrices vs collapsed CVFSNet fuse01 baselines:**
```bash
uv run python -m mae_ordinal.plot_confusion_binary
```
→ `plots/confusion_matrices_binary.png`

| VideoMAE run | CVFSNet baseline | Collapse |
|--------------|------------------|----------|
| `mae_binary_t012a_ap` | `cvfsnet_pt_fuse01_cor_v1` | fuse01 0,1→0; 2,3→1 |
| `mae_binary_t012a_sag` | `cvfsnet_pt_fuse01_sag_v1` | same |
| `mae_binary_t012a_dual` | `cvfsnet_pt_fuse01_fusion_v1` | same |

Baseline comparison uses **post-hoc collapsed fuse01 predictions**, not BCE-trained binary CVFSNet.

---

## 6. New / modified files

| File | Purpose |
|------|---------|
| `mae_ordinal/model.py` | `VideoMAEBinary`, `VideoMAEBinaryDual`, `build_videomae_binary()` |
| `mae_ordinal/binary_lightning_module.py` | BCE Lightning module |
| `mae_ordinal/dataloader.py` | AP / sagittal / dual binary presets |
| `mae_ordinal/train.py` | Routes binary vs ordinal by `model.name` |
| `mae_ordinal/config_binary_*.yaml` | Three experiment configs |
| `mae_ordinal/eval_binary.py` | Checkpoint eval + sigmoid CSV export |
| `mae_ordinal/plot_confusion_binary.py` | Binary CM comparison plot |
| `mae_ordinal/docs/BINARY_RESULTS_REPORT.md` | Results summary (populated after training) |

---

## 7. Caveats

- No held-out test set (`test=val`); numbers are val-set only.
- CVFSNet baselines are 4-class fuse01 models collapsed to binary for comparison.
- Dual-view uses batch size 4 to avoid OOM on 24 GB GPUs.
