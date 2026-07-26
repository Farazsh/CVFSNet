# DINOv3 binary mTICI experiment

This package fine-tunes a selectable DINOv3 ViT-S/16 backend on AP DSA studies.
Each NIfTI volume is trilinearly resampled to three chronological
frames; early, middle, and late frames become the three channels of one
224 x 224 image.

The default is Lightly Train's public `dinov3/vits16` model. Meta's gated
`facebook/dinov3-vits16-pretrain-lvd1689m` Hugging Face model remains available:

```bash
# Lightly Train (default; no Hugging Face token required)
uv run python -m dinov3.train --set model.backend=lightly

# Hugging Face
uv run python -m dinov3.train --set model.backend=huggingface \
  run.name=dinov3_hf_vits16_binary_tici_ap_seed14207
```

For the Hugging Face backend, accept the gated model terms and place a token
from that account in the repository `.env` as `HF_TOKEN`. The token is read at
runtime and is not written to configs, logs, or checkpoints.

```bash
uv sync
uv run python -m dinov3.train
```

Configuration can be overridden without editing the default YAML:

```bash
uv run python -m dinov3.train --set \
  trainer.max_epochs=2 \
  logging.wandb.enabled=false \
  data.loader.num_workers=0 \
  data.loader.val_num_workers=0
```

To re-evaluate an existing checkpoint:

```bash
uv run python -m dinov3.evaluate path/to/best.ckpt
```

The best checkpoint is selected by validation AUROC, reloaded into a fresh
model, and evaluated on the 150-study validation cohort. The decision threshold
maximizes macro F1 on that same cohort. Outputs include per-study scores,
evaluation JSON, and an Excel workbook matching the supplied reference schema.
Because validation selects the checkpoint and threshold and is also reported,
these metrics are comparison results rather than held-out performance estimates.
