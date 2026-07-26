# VideoMAE / VideoMAE2 binary mTICI experiments

Binary T012a vs T2b3 classification on AmTICIS DSA clips. Training reuses
`amticis_pipeline` (16 frames, 224×224). Epoch metrics and post-fit artifacts
match the `dinov3` contract: flat `val/auroc` (checkpoint monitor),
macro-F1 threshold tuning, `val_scores.csv`, `evaluation.json`, and Excel.

## Requirements

- `uv sync` (includes `transformers` and `timm` for VideoMAE2 remote code)
- VideoMAE2 loads `OpenGVLab/VideoMAEv2-Base` with `trust_remote_code=true`
  (weights download into the Hugging Face cache on first use). Requires
  `timm` and `easydict`. Hub calls read `HF_TOKEN` from the repo `.env` for
  authenticated (faster) downloads. Loading uses `from_config` + safetensors
  for transformers compatibility.

## Binary training (VideoMAE v1)

```bash
uv run python -m mae_ordinal.train --config mae_ordinal/config_binary_ap.yaml
uv run python -m mae_ordinal.train --config mae_ordinal/config_binary_sagittal.yaml
uv run python -m mae_ordinal.train --config mae_ordinal/config_binary_dual.yaml
```

## Binary training (VideoMAE2)

Configs default to **4-GPU DDP** (`devices: [0,1,2,3]`) with a small per-GPU
batch (AP/sag: 2, dual: 1) to avoid 24 GB OOM.

To use a non-16 frame length (e.g. 8), set both
`data.pipeline_overrides.data.num_frames` and `model.params.expected_num_frames`
to the same even value. The loader regenerates VideoMAE2 sin-cos `pos_embed`
for the new tubelet count (`T` must be divisible by `tubelet_size=2`).

```bash
uv run python -m mae_ordinal.train --config mae_ordinal/config_binary_v2_ap.yaml
uv run python -m mae_ordinal.train --config mae_ordinal/config_binary_v2_sagittal.yaml
uv run python -m mae_ordinal.train --config mae_ordinal/config_binary_v2_dual.yaml
```

4-GPU smoke check (1 epoch, few batches, no W&B):

```bash
uv run python -m mae_ordinal.train --config mae_ordinal/config_binary_v2_ap.yaml --set \
  run.name=mae_v2_binary_t012a_ap_smoke \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=2 \
  logging.wandb.enabled=false \
  checkpointing.save_last=false
```

Smoke-load the VideoMAE2 backbone (download + forward; uses `.env` `HF_TOKEN`):

```bash
uv run python -c "
import torch
from mae_ordinal.model import VideoMAEv2Binary, resolve_hf_token
assert resolve_hf_token(), 'HF_TOKEN missing from .env'
m = VideoMAEv2Binary()
print(m(torch.randn(1, 1, 16, 224, 224)).shape)
"
```

## Evaluation

After training (also runs automatically for binary fits):

```bash
uv run python -m mae_ordinal.eval_binary --run mae_v2_binary_t012a_ap
```

## Ordinal (unchanged)

```bash
uv run python -m mae_ordinal.train --config mae_ordinal/config_frozen.yaml
uv run python -m mae_ordinal.train --config mae_ordinal/config_finetune.yaml
```

Ordinal still monitors `val/fuse/f1_macro` (CORN). Binary monitors `val/auroc`.
