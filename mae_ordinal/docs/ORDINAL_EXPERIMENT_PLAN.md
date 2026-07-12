# VideoMAE Ordinal Regression Experiment Plan

**Status:** Draft for review. **No implementation code has been written.** Do not start
building until this plan is explicitly approved.

**Branch:** `mae-ordinal-coronal` (created; existing CVFSNet code untouched).

**Goal:** Test whether reformulating mTICI grading as **ordinal regression** on top of a
**VideoMAE** backbone beats the existing **coronal-only CVFSNet CNN** baseline
(`cvfsnet_pt_fuse01_cor_v1`), single view (coronal) only, using the exact same
train/val split, label scheme, and evaluation metrics so results are directly comparable.

### Locked decisions (from review, 2026-07-11)

These are now fixed for the implementation — no longer open questions:

1. **Data loader:** use the `amticis_pipeline` package's `AmTICISDataModule` (the current
   Lightning pipeline). Do **not** use the older `Data/dataset.py` loader.
2. **View:** **AP (coronal) only** — single view, no sagittal, no dual-branch duplication.
   Use the Lightning module's `single_view_tensor` input path.
3. **Preprocessing:** keep the existing `ResizeView` **trilinear** interpolation, but
   resample to **16 frames** at **224×224** (VideoMAE's expected clip geometry).
4. **Labels:** `fuse01` (**4 ordinal classes**) as the ordinal-regression target.
5. **Imbalance:** reuse the existing `WeightedRandomSampler` (inverse-frequency oversampling).
6. **Metrics:** reuse the existing `ClassificationMetricAccumulator`; primary key
   `val/fuse/f1_macro` + `val/fuse/accuracy`, same split/seed, `test=val`.
7. **Dependencies:** install packages as needed (`transformers`, and download the VideoMAE
   checkpoint).
8. **Normalization (updated):** use the **dataset_loader's own normalization**
   (`TioZNormalization(div255=true)`) exactly as the existing pipeline produces it — **no
   ImageNet stats**. The grayscale z-normalized clip is simply channel-replicated 1→3 and fed to
   VideoMAE; fine-tuning adapts the backbone to this input distribution.
9. **Fine-tuning (updated):** **full fine-tuning — no frozen parameters.** All VideoMAE weights
   train. Use a single small learning rate suitable for full transformer fine-tuning on a tiny
   dataset: **`lr = 1e-5`, AdamW, weight_decay 0.05, cosine schedule with short warmup, grad
   clip**. (No freeze/unfreeze phases.)
10. **Experiment tracking:** **Weights & Biases enabled for all runs** (project `AmTICIS`); the
    API key is stored in `.env` (git-ignored) and read via `WANDB_API_KEY`.

---

## STEP 1 — Investigation of the existing repo

The repo contains **two** pipelines:

1. **Legacy paper pipeline** — `main.py`, `config.py`, `Src/CVFSNet.py`, `Data/`, `Loss/`,
   `Lib/` (yacs config, custom train loop). This is the original release.
2. **Active Lightning pipeline** — `amticis_pipeline/` (data) + `amticis_training/`
   (model/loss/metrics/trainer). **This is what you reproduced and ablated**, and is the
   pipeline the new experiment will plug into. All citations below refer to it unless noted.

### 1.1 Dataset / dataloader — how coronal frames are loaded

- **DataModule:** `amticis_pipeline/dataset_loader.py::AmTICISDataModule` (a
  `LightningDataModule`). `setup()` reads the split JSON, builds train/val/test
  `AmTICISDataset`s, and wires transforms. Dataloaders: `train_dataloader`,
  `val_dataloader`, `test_dataloader`.
- **Dataset:** `amticis_pipeline/dataset.py::AmTICISDataset`. For each study it loops over
  the active `ViewSpec`s and calls `_load_view(base_name, view)`:
  - `amticis_pipeline/utils.py::load_nifti()` → `SimpleITK.GetArrayFromImage` → float32
    `(H, W, T)` numpy array (raw DSA contrast preserved, **no windowing** at load time).
  - `amticis_pipeline/transforms.py::ResizeView.__call__` → a single 3D **trilinear**
    `F.interpolate` that resamples the whole `(H, W, T)` clip to a fixed
    `(1, num_frames, image_size, image_size)` tensor (channel-first, `C=1`). **This is where
    "frame selection" happens** — there is no discrete frame sampling/striding; an arbitrary
    native frame count is resampled temporally to `num_frames` (currently **8**).
  - The configured transform pipeline (`ComposeTransforms`) is then applied.
- **Output per sample** (`AmTICISDataset.__getitem__`): a dict
  `{ "AP": tensor(1, T, H, W), ["sagittal": ...], "label": long tensor shape (1,), "name": str }`.
  For our experiment only the coronal view (`"AP"`) is active.

**Subsample/select / resize / normalize / augment (exact steps, from
`amticis_pipeline/config.yaml` → resolved through `amticis_pipeline/transforms.py`):**

- **Resize/select:** `ResizeView` (trilinear, `align_corners=true`) → `(1, 8, 256, 256)`.
- **Train augmentations** (declarative list in `amticis_pipeline/config.yaml`, classes
  reused verbatim from `Data/transforms.py` via `transforms._instantiate_step`, except
  `RandomErode`/`RandomDilate` which have OpenCV-5-safe local overrides in
  `amticis_pipeline/transforms.py::_Morphology`):
  `RandomErode` (p=.15), `RandomDilate` (p=.15), `TioClamp` (p=0, i.e. **off**),
  `TioRandomFlip` (p=.25), `TioRandomAnisotropy`, `TioRandomMotion`, `TioRandomGhosting`,
  `TioRandomSpike`, `TioRandomBiasField`, `TioRandomBlur`, `TioRandomNoise`, `TioRandomGamma`
  (all p=.25), `RandomRotation` (±30°, fill=192, p=.25), `Crop` (p=.5),
  `Resize` (re-fixes T/spatial after crop; `t`/`visual` injected from `data.*`),
  and finally `TioZNormalization(p=1, div255=true)`.
- **Val/test augmentations:** deterministic — only `TioZNormalization(p=1, div255=true)`.
- **Normalization detail (`Data/transforms.py::TioZNormalization`):** applies
  `torchio.ZNormalization()` (per-volume **zero-mean / unit-std** over the whole clip), then
  divides by 255. Result: single-channel, per-volume standardized, then `/255` (so values are
  small; not an ImageNet-style normalization). **This is important for the VideoMAE
  reconciliation — see §2.1.**
- **Class-balanced sampling:** `train_dataloader` uses a `WeightedRandomSampler` with
  inverse-frequency weights from `AmTICISDataset.sample_weights()` when
  `loader.use_weighted_sampler=true` (it is).

### 1.2 TICI/mTICI labels — raw set and encoding

- **Raw grade tokens** parsed from the file name (`utils.parse_label_token`:
  `Path(name).name.split("_")[1].upper()`): scans are named e.g. `0007_T2A_C_anon.nii.gz`,
  so tokens are `T0, T1, T2A, T2B, T3`. (No `T2C` token exists in this dataset.)
- **Label maps** (`amticis_pipeline/utils.py`):
  - `full`  → 5 classes: `T0→0, T1→1, T2A→2, T2B→3, T3→4`
  - `fuse01` → **4 classes**: `T0→0, T1→0, T2A→1, T2B→2, T3→3` (T0+T1 merged) — **the paper's
    setting and the one used by the coronal-only baseline.**
  - `binary` → 2 classes: `{T0,T1,T2A}→0, {T2B,T3}→1`
- Selected via `data.label_mode` in `amticis_pipeline/config.yaml` (overridable from
  `amticis_training/config.yaml → data.pipeline_overrides.data.label_mode`).
  `num_classes` derives automatically (`amticis_pipeline/config.py::NUM_CLASSES_BY_MODE`).
- **Model target:** `label` is a `(1,)` long tensor; the Lightning module flattens to
  `(B,)` (`lightning_module._shared_step`). **For our experiment N = 4 (fuse01).** These are
  genuinely ordinal: `T0/1 < T2a < T2b < T3` (increasing reperfusion quality).

### 1.3 Coronal-only ablation — how it is switched on

Confirmed from saved run `output_runs_lightning/cvfsnet_pt_fuse01_cor_v1/resolved_training_config.yaml`
(this is the coronal-only fuse01 CVFSNet baseline we must beat):

- `data.pipeline_overrides.views.active: [AP, sagittal]` **but both view definitions map to the
  coronal file**: `AP: {_C→_C}` and `sagittal: {_C→_C}`. i.e. CVFSNet still runs its two
  branches, but **both branches receive the coronal clip** — an "coronal-only" ablation that
  keeps the dual-view architecture. `label_mode: fuse01`, `model.params.model_num_class: 4`.
- (Sagittal-only and true dual-view use `_C→_S` for the sagittal definition; see
  `cvfsnet_pt_fuse01_sag_v1` / `cvfsnet_pt_fuse01_fusion_v1`.)
- The Lightning module also supports a genuinely single-view input path:
  `lightning_module._batch_to_model_input` handles `input.type: single_view_tensor` (returns
  one view tensor). **Our VideoMAE model will use this single-view path** with
  `views.active: [AP]`, `AP: {_C→_C}` — strictly one coronal clip per sample, no duplicated
  branch.

### 1.4 Model entry point, training loop, config, logging/checkpoint conventions

- **Entry point:** `python -m amticis_training` (`amticis_training/__main__.py` → `train.main`)
  or `python -m amticis_training.train`. CLI: `--config`, `--set dotted.key=value`,
  `--print-config`.
- **Config system:** `amticis_training/config.py::load_training_config` loads
  `amticis_training/config.yaml`, deep-merges `--set` overrides. Data config is a **separate**
  YAML (`amticis_pipeline/config.yaml`) referenced via `data.config_path`, with
  `data.pipeline_overrides` deep-merged (`build_pipeline_config`). Required top-level sections:
  `data, loss, metrics, model, optimizer, run, trainer`.
- **Model registry:** `amticis_training/models.py::MODEL_REGISTRY` + `@register_model(name)`;
  `build_model(config["model"], pipeline_config)` dispatches on `model.name`. **We add a new
  entry `videomae_ordinal` here from our package — without editing existing builders.** (The
  registry is a dict; we import `amticis_training.models` and register into it, or expose our
  own `build_model`; see §2.)
- **Loss registry:** `amticis_training/losses.py::LOSS_REGISTRY` + `@register_loss(name)`.
- **Training loop:** `amticis_training/lightning_module.py::AmTICISLightningModule`.
  Key behaviors we must satisfy:
  - `forward(model_input)`; input built by `_batch_to_model_input` (we use
    `single_view_tensor`).
  - `_unwrap_outputs` expects the model to return a **list/tuple of tensors** (or a single
    tensor); `criterion(outputs, label)` must return `(loss, loss_dict)`.
  - Metrics: `_record_metrics` feeds each `(B, num_classes)` output tensor to a
    `ClassificationMetricAccumulator` keyed by `model.output_names[i]`.
- **Trainer:** `amticis_training/train.py::main` builds a Lightning `Trainer(**config["trainer"])`
  with CSV + optional W&B loggers, `ModelCheckpoint`, optional `EarlyStopping`,
  `LearningRateMonitor`.
- **Run/output dir:** `output_runs_lightning/<run.name>/`; resolved config dumped to
  `resolved_training_config.yaml`; CSV logs under `csv/`; checkpoints via `ModelCheckpoint`
  (`monitor: val/fuse/f1_macro`, `mode: max`, `save_top_k`, `save_last`, filename
  `epoch={epoch:03d}`).
- **Seeding:** `seed_everything(config.seed, workers=True)` (default 14207).

### 1.5 Evaluation metrics already computed (must be matched exactly)

`amticis_training/classification_metrics.py::ClassificationMetricAccumulator` accumulates
epoch-level, one-vs-rest metrics from `(N, num_classes)` softmax probabilities + integer
targets. Available metric names: **accuracy, auroc, auprc, precision, recall, f1,
specificity, tp, tn, fp, fn** (+ per-class variants and `*_macro` / `*_total`).

- Logged keys look like `val/<output_name>/<metric>`, e.g. `val/fuse/f1_macro`,
  `val/fuse/accuracy`, `val/fuse/class_2/recall`.
- **Primary comparison metric = `val/fuse/f1_macro`** (this is the checkpoint/early-stop
  `monitor` for the baseline) plus `val/fuse/accuracy`. The coronal baseline also logs `AP`
  and `sagittal` outputs; since both are coronal there, `fuse` is the head to compare against.
- Note: this accumulator does **not** compute MCC or Cohen's kappa. The legacy
  `Lib/metric.py::get_metrics` computes extras (`acc_smooth` = off-by-one accuracy, macro
  precision/recall/f1, AUC) but is **not** used by the Lightning pipeline. To stay
  "directly comparable" we will reuse `ClassificationMetricAccumulator` verbatim and report the
  **same keys**; ordinal-specific metrics (QWK, MAE, off-by-one acc) are added as **extra**
  logs, clearly separated (see §2.3, open question Q4).

### 1.6 Dataset size, splits, class distribution

From `train_val_split.json` (single top-level key `CORONAL_VIEW`; base names are `_C` coronal):

- **Splits:** `train = 261`, `val = 150`. **No dedicated test split** — `amticis_pipeline/config.yaml`
  sets `splits.test: val` (test reuses val). The baseline was evaluated on `val`.
- **Class distribution (fuse01, 4-class):**

  | split | 0: T0/T1 | 1: T2a | 2: T2b | 3: T3 | total |
  |-------|---------:|-------:|-------:|------:|------:|
  | train | 98 | 30 | 48 | 85 | 261 |
  | val   | 51 | 29 | 23 | 47 | 150 |

  Raw (train): T0=80, T1=18, T2a=30, T2b=48, T3=85. Imbalanced (T2a is the minority in train;
  class 0 dominates). Handled today by the `WeightedRandomSampler` + seesaw loss.

### 1.7 Dependencies, torch/CUDA, GPU memory

- **Dep files:** `pyproject.toml` (uv-managed, `package=false`) + `uv.lock`. Torch pinned
  `torch==2.4.1`, `torchvision==0.19.1` from the `cu121` index. Env: `.venv/` (Python 3.11, uv).
- **Installed:** `torch 2.4.1+cu121` (CUDA 12.1 runtime), `pytorch-lightning>=2.6`, sklearn,
  torchio, SimpleITK, opencv, wandb, etc. **`transformers` is NOT installed** and **`timm` is
  NOT installed** — both must be added (see §2 / risks).
- **GPUs:** 4× **NVIDIA RTX 3090, 24 GB each**, driver 595.71 / CUDA 13.2 (backward-compatible
  with the cu121 build), all idle. **Plenty of memory** — VideoMAE-Base fine-tuning at 16
  frames / 224² is comfortable on a single 3090; multi-GPU DDP is available but unnecessary
  for a dataset this small.

**Feasibility verdict:** VideoMAE-**Base** (≈86M params) is feasible with room to spare;
VideoMAE-**Large** (≈305M) would fit memory-wise but is almost certainly too large for 261
training samples. See §2.2.

---

## STEP 2 — Design of the new `mae_ordinal/` package (NOT yet built)

**Hard constraint:** do not modify any file inside `amticis_pipeline/`, `amticis_training/`,
`Src/`, `Data/`, `Loss/`, `Lib/`. Only **import** from them.

### Proposed package layout

```
mae_ordinal/
  __init__.py
  dataloader.py     # wraps amticis_pipeline output for VideoMAE
  model.py          # VideoMAE backbone + ordinal (CORN) head
  ordinal.py        # CORN loss + logits→rank / →class-prob conversion (small, self-contained)
  metrics.py        # thin adapter: reuse ClassificationMetricAccumulator + extra ordinal metrics
  lightning_module.py  # subclass/compose amticis_training module OR a small dedicated module
  train.py          # entry point (mirrors amticis_training/train.py conventions)
  config.yaml       # experiment config (mirrors amticis_training/config.yaml structure)
  README.md
```

### 2.1 `dataloader.py` — reuse existing preprocessing, adapt to VideoMAE

**Reuse (import, do not reimplement):** `amticis_pipeline.dataset_loader.AmTICISDataModule`,
`amticis_pipeline.config.load_config`/`PipelineConfig`, and therefore transitively
`ResizeView`, `build_transform_pipeline`, `AmTICISDataset`, and the `Data/transforms.py`
augmentations. Plan: instantiate `AmTICISDataModule` with a `PipelineConfig` that has
`views.active=[AP]`, `AP:{_C→_C}`, `label_mode=fuse01`, and then apply a **thin post-transform
adapter** that converts the pipeline's `(1, T, H, W)` clip into the format the VideoMAE model
consumes. Two clean options for where the adapter lives (recommend **A**):

- **A. Adapter inside the model's forward** (dataloader stays 100% the stock pipeline output).
  The Lightning `single_view_tensor` path already yields `batch["AP"]` of shape `(B,1,T,H,W)`;
  the model's `forward` reshapes/repeats channels and resizes to VideoMAE's expectations. This
  keeps `dataloader.py` a very thin config-builder and avoids touching the collate path.
- **B. Adapter as a `collate_fn`/wrapping `Dataset`** in `mae_ordinal/dataloader.py` that
  post-processes each sample. More explicit but duplicates some tensor plumbing.

Recommendation: **A** — `dataloader.py` just builds the `AmTICISDataModule` with the coronal
single-view `PipelineConfig`; the VideoMAE-specific tensor adaptation happens in `model.py`.

**VideoMAE processor expectations vs. our pipeline (the conflicts to reconcile):**

| Aspect | VideoMAE default (`VideoMAEImageProcessor` / K400 ckpt) | Our pipeline output | Reconciliation |
|---|---|---|---|
| Frame count | **16** (tubelet_size 2 → 8 temporal tokens) | 8 | **LOCKED: `data.num_frames: 16`** via pipeline override (ResizeView resamples temporally — free to change). Keeps positional-embedding shape valid. |
| Spatial size | **224×224** | 256×256 | **LOCKED: `data.image_size: 224`** via ResizeView. Matches pretrained weights exactly — no pos-emb interpolation needed. |
| Channels | **3 (RGB)** | **1 (grayscale)** | **Repeat** the single channel to 3 (`x.repeat/expand` on the channel dim). DSA is grayscale; channel-replication is the standard, information-preserving choice. |
| Normalization | ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, on `[0,1]` pixels | `torchio ZNormalization` then `/255` (per-volume z-score, tiny magnitude) | See decision below. |
| dtype/layout | `pixel_values (B, T=16, C=3, H, W)` | `(B, 1, T=16, H, W)` | Permute `(B,1,T,H,W)→(B,T,1,H,W)`, repeat C→3. |

**Normalization decision (FINAL, per review):** Use the **dataset_loader's normalization
unchanged** — `TioZNormalization(div255=true)`, i.e. per-volume z-score then `/255`. We do
**not** apply ImageNet stats. The model adapter only permutes and channel-replicates (1→3); it
does not renormalize. This keeps preprocessing byte-for-byte identical to the CVFSNet coronal
baseline (fairest comparison); the fully fine-tuned VideoMAE adapts to this input distribution.

**Augmentation note (flagged, Q2):** the DSA-specific torchio augmentations operate on the
grayscale clip *before* channel replication — fine. But several assume the pipeline's intensity
range; after switching normalization we keep augmentations *before* the final
normalization/ImageNet step so their assumptions hold. `TioClamp` is already p=0 (off).

### 2.2 `model.py` — VideoMAE backbone + ordinal head

- **Library / checkpoint:** `transformers` `VideoMAEModel` (backbone, no classifier) from
  **`MCG-NJU/videomae-base-finetuned-kinetics`** (Base, Kinetics-400 fine-tuned).
  - **Base vs Large:** **Base (~86M)**. With only 261 training clips, Large (~305M) will
    overfit and is not justified; Base is the standard choice for small downstream video sets.
  - **Which pretrained source:** **Kinetics-400 fine-tuned** rather than the raw SSL
    (`videomae-base`) checkpoint or Kinetics-710. Rationale: the K400-fine-tuned encoder yields
    more semantically organized features that transfer better when we freeze most of the
    backbone and train only a small head on a tiny dataset; SSL-only features would need more
    unfreezing/data. (Kinetics-710 checkpoints are mainly VideoMAE **v2**; see below.)
  - **VideoMAE v2:** v2 checkpoints (`OpenGVLab/VideoMAEv2-*`) generally require custom modeling
    code / `trust_remote_code` and are less turn-key in `transformers`. **Proposed: use v1 Base
    for a clean, maintained path**; v2 is a stretch-goal only if v1 underperforms (Q3).
- **Head:** drop the classification head; take the backbone's sequence output, mean-pool tokens
  (or use the pooled/CLS-equivalent representation) → a small MLP → **ordinal head** producing
  `K−1 = 3` outputs (for N=4).
- **Ordinal formulation:** **CORN** (Conditional Ordinal Regression for Neural networks,
  Shi/Cao/Raschka 2021) over **CORAL**. Rationale: CORN uses conditional probabilities and does
  **not** impose CORAL's shared-weight / rank-monotonicity constraint, so it is strictly more
  flexible and empirically ≥ CORAL, while still guaranteeing rank-consistent predictions at
  inference. It needs no extra learned bias-ordering constraints. Implementation is small and
  will live in `mae_ordinal/ordinal.py` (self-contained; no new heavyweight dependency — we do
  **not** need `coral-pytorch`, though it can be a reference).
  - Output: `(B, K−1)` logits. Rank prediction = number of thresholds with prob>0.5 (the
    standard rank-consistent CORN decoding).

### 2.3 `train.py` + `config.yaml`

- **Fine-tuning schedule (FINAL, per review): FULL fine-tuning, nothing frozen.**
  - All VideoMAE parameters + head are trainable from step 0.
  - Single LR **`1e-5`**, **AdamW**, `weight_decay=0.05`, **cosine schedule** with a short
    warmup, **gradient clipping** (`1.0`). A low LR is essential when fully fine-tuning an ~86M
    transformer on 261 clips; strong augmentation + weighted sampling + checkpointing on
    `val/fuse/f1_macro` mitigate overfitting.
  - Keep the rest of the training conventions identical to `amticis_training` (seed, CSV + W&B
    logging, `ModelCheckpoint` on the primary metric).
- **Loss:** CORN loss (from `ordinal.py`). Wrapped to return `(loss, loss_dict)` so it plugs
  into the Lightning module contract. `MultiOutputCrossEntropy`-style multi-head weighting is
  not needed (single head).
- **Class imbalance:** reuse the existing **`WeightedRandomSampler`** (already in the data
  module, inverse-frequency) — this is the baseline's mechanism, keeping the comparison fair.
  Optionally add per-threshold loss weighting in CORN; default off unless val shows minority
  collapse.
- **Metrics / eval protocol (identical to baseline):** same train/val split, `test=val`, same
  `seed`. To feed the repo's `ClassificationMetricAccumulator` (which expects
  `(N, num_classes)` scores), `mae_ordinal/metrics.py` converts CORN cumulative outputs → a
  proper **4-class probability vector** per sample (`P(y=k)` from the cumulative link), then
  passes those as the "logits" so **the exact same metric definitions** (accuracy, macro-F1,
  precision/recall/specificity, AUROC/AUPRC, per-class) are computed and logged under
  `val/fuse/*` for apples-to-apples comparison with `cvfsnet_pt_fuse01_cor_v1`.
  - We name the single output `fuse` so the primary key is literally `val/fuse/f1_macro`,
    matching the baseline's checkpoint monitor.
  - **Extra ordinal metrics** (logged separately, e.g. `val/fuse/qwk`, `val/fuse/mae`,
    `val/fuse/acc_off_by_one`) since they are the natural payoff of ordinal modeling — but they
    are additive, not a substitute.
- **Lightning module:** prefer to **reuse `amticis_training.lightning_module.AmTICISLightningModule`
  unchanged** by making our model return `[class_prob_logits]` and our loss accept the CORN
  target — *if* that fits cleanly. If the CORN loss needs the raw `(B,K−1)` head output while
  metrics need `(B,N)` class probs, we will instead add a small dedicated
  `mae_ordinal/lightning_module.py` that composes the same `ClassificationMetricAccumulator` and
  the same logging keys (no edits to the existing module). Decision recorded as Q5.

### 2.4 Registration without touching existing files

`amticis_training.models.MODEL_REGISTRY` / `losses.LOSS_REGISTRY` are module-level dicts with
`register_*` decorators. Our `mae_ordinal` package will **import** those modules and register
`videomae_ordinal` / `corn` into them at import time (side-effect registration), OR our own
`train.py` will build the model/loss directly and pass them to the Lightning module. Either way,
**no file under `amticis_training/` is edited.** (Recommend the direct-build path in our own
`train.py` for clarity.)

---

## Key design decisions & justification (summary)

| Decision | Choice | Why |
|---|---|---|
| Backbone | VideoMAE-**Base**, `transformers` `VideoMAEModel` | 261 train samples; Large would overfit; Base is standard for small video sets. |
| Pretrained source | `MCG-NJU/videomae-base-finetuned-kinetics` (K400) | Semantically organized features transfer best under heavy freezing; SSL-only needs more data/unfreezing. |
| Frame count | 16 (from 8) | Matches pretrained temporal positional embeddings; ResizeView resamples for free. |
| Resolution | 224 (from 256) | Matches pretrained spatial pos-emb exactly; avoids pos-emb interpolation. |
| Channels | grayscale→3 by replication | DSA is single-channel; replication is lossless & standard. |
| Normalization | ImageNet stats via model adapter (replace final norm only) | VideoMAE expects ImageNet-normalized RGB; existing z-norm/÷255 is incompatible. Flagged, not silent. |
| Ordinal head | **CORN** (K−1=3 outputs) | Rank-consistent, more flexible than CORAL, small self-contained impl. |
| Freeze schedule | freeze-all → unfreeze top 2–4 blocks + head, discriminative LR | Standard low-data transfer recipe; avoids catastrophic forgetting. |
| Imbalance | existing `WeightedRandomSampler` | Same mechanism as baseline → fair comparison. |
| Eval | reuse `ClassificationMetricAccumulator`, key `val/fuse/*`, `test=val`, same split/seed | Directly comparable to `cvfsnet_pt_fuse01_cor_v1`. |

---

## Open questions / risks / uncertainties

**Resolved by review:** data loader (`amticis_pipeline`), AP-only, 16 frames / 224², fuse01,
`WeightedRandomSampler`, existing metrics, install-as-needed. Normalization → ImageNet stats via
adapter (see §2.1). The only items still worth a decision:

- **Q3 (v1 vs v2):** Proceed with VideoMAE **v1** Base (maintained in `transformers`); attempt
  VideoMAE **v2** only as a follow-up if v1 underperforms. **Default: v1 first** unless you object.
- **Q5 (module reuse, internal):** Reuse `AmTICISLightningModule` if the CORN head fits its
  list-of-logits contract; otherwise add a small dedicated module in `mae_ordinal/` (no edits to
  existing files). This is an implementation detail; I'll pick whichever is cleaner.

**Risks:**

- **No held-out test set:** baseline used `test=val`; we mirror it for comparability, but
  val-tuned numbers are optimistic (consider a proper test split later; out of scope now).
- **Tiny dataset + big transformer:** high overfitting risk; mitigated by heavy freezing, strong
  augmentation, and checkpointing on `val/fuse/f1_macro`.
- **New deps / checkpoint download:** adding `transformers` + first-time HF download of
  `MCG-NJU/videomae-base-finetuned-kinetics` (~330 MB) needs network access on this machine
  (or a manual `hf download`). CUDA 13.2 driver is backward-compatible with the cu121 torch build.
- **CORN→4-class prob mapping for AUROC/AUPRC:** cumulative-link→categorical conversion is
  standard but must be sanity-checked in the smoke test (per-class AUROC from 3 thresholds).

---

## STEP 3 — Implementation checklist (execute only after approval)

1. **Env:** add `transformers` to `pyproject.toml`; `uv sync`; verify `.venv` imports
   `transformers` + downloads `MCG-NJU/videomae-base-finetuned-kinetics` (or pre-download).
2. **Scaffold** `mae_ordinal/` package (files per layout above), `__init__.py` exports.
3. **`dataloader.py`:** build a coronal single-view `PipelineConfig`
   (`views.active=[AP]`, `AP:{_C→_C}`, `label_mode=fuse01`, `num_frames=16`, `image_size=224`,
   final-normalization override per Q1) and instantiate `AmTICISDataModule`. Assert output
   shapes/dtype.
4. **`ordinal.py`:** implement CORN loss + `logits→rank` + `logits→4-class prob` helpers; unit
   test on toy tensors (monotonicity, correct rank decoding).
5. **`model.py`:** load `VideoMAEModel` Base; adapter (permute, channel-replicate, ImageNet
   norm); pooling + MLP + CORN head; freeze-all initially; return `[class_prob_logits]` (+ raw
   CORN output for the loss). Parameter groups for discriminative LR.
6. **`metrics.py`:** adapter that reuses `ClassificationMetricAccumulator` on the 4-class probs;
   add QWK/MAE/off-by-one.
7. **`lightning_module.py`** (only if needed per Q5) + **`train.py`** mirroring
   `amticis_training/train.py` (CSV/W&B, `ModelCheckpoint` monitor `val/fuse/f1_macro`, seed,
   cosine schedule, grad clip). New `config.yaml`.
8. **SMOKE TEST (before any full run):** `--set trainer.max_epochs=1 trainer.limit_train_batches=2
   trainer.limit_val_batches=2` on ~a couple of samples: confirm data shapes, forward pass, loss
   is finite and decreases on an overfit-2-batches check, metrics log under `val/fuse/*`, and a
   checkpoint is written. Verify the frozen-parameter count and that only head/top-blocks have
   grads.
9. **Phase-1 short run** (head only, ~10–20 epochs) → sanity of `val/fuse/f1_macro`.
10. **Phase-2 fine-tune** (unfreeze top blocks, discriminative LR) → full training with the same
    early-stop/checkpoint protocol as the baseline.
11. **Compare** `val/fuse/{f1_macro,accuracy,...}` against `cvfsnet_pt_fuse01_cor_v1`; report the
    same keys + ordinal extras.

---

## Approximate implementation time

Coding/scaffolding effort (not counting GPU training wall-clock):

| Task | Est. |
|---|---|
| Env: add `transformers`, `uv sync`, pre-download checkpoint | ~15–30 min (mostly download) |
| `dataloader.py` (coronal single-view config + adapter) | ~30–45 min |
| `ordinal.py` (CORN loss + rank/prob decoders + unit test) | ~45–60 min |
| `model.py` (VideoMAE load, adapter, pooling+CORN head, freeze logic, param groups) | ~1–1.5 h |
| `metrics.py` (reuse accumulator + QWK/MAE/off-by-one) | ~30–45 min |
| `train.py` + `config.yaml` (+ dedicated Lightning module if needed) | ~1 h |
| Smoke test + debugging shapes/wiring | ~30–60 min |

**Total hands-on coding: roughly 4–6 hours** to a working, smoke-tested experiment.

**Training wall-clock (separate, runs unattended):** VideoMAE-Base, mostly frozen, 261 clips at
16×224², batch ~8–16 on one RTX 3090 → on the order of a few seconds per step; a full run
(phase-1 head warm-up + phase-2 partial fine-tune, with checkpointing) is typically **~1–3 hours**
depending on epochs/early-stopping. First run adds the one-time checkpoint download.

## STEP 4 — Awaiting review

**I will not write any model/dataloader/training code until you approve this plan.** Your review
decisions are locked in above. Remaining minor calls: Q3 (v1 first) and Q5 (module reuse, my
discretion). Approve to proceed, or flag the normalization default in §2.1 if you'd prefer raw
z-normalized inputs instead.
