"""Regenerate val-set predictions from saved checkpoints and plot confusion
matrices for the two VideoMAE runs (frozen / fine-tune) alongside the CVFSNet
coronal baseline.

    uv run python -m mae_ordinal.plot_confusion

Outputs: plots/confusion_matrices.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import cohen_kappa_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["T0/1", "T2a", "T2b", "T3"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUNS = {
    "VideoMAE frozen": "mae_ordinal_fuse01_ap_frozen",
    "VideoMAE fine-tune": "mae_ordinal_fuse01_ap_finetune",
    "CVFSNet coronal (baseline)": "cvfsnet_pt_fuse01_cor_v1",
}


def _run_dir(name: str) -> Path:
    return REPO_ROOT / "output_runs_lightning" / name


def _best_ckpt(name: str) -> Path:
    ckpts = sorted((_run_dir(name) / "csv").glob("version_*/checkpoints/epoch=*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No best checkpoint for {name}")
    return ckpts[-1]


def _resolved_config(name: str) -> dict:
    with open(_run_dir(name) / "resolved_training_config.yaml") as handle:
        return yaml.safe_load(handle)


@torch.no_grad()
def _predict_videomae(run: str) -> Tuple[np.ndarray, np.ndarray]:
    from .dataloader import build_datamodule
    from .model import build_videomae_ordinal
    from .ordinal import corn_class_probs

    config = _resolved_config(run)
    dm = build_datamodule(pipeline_overrides=config["data"].get("pipeline_overrides"))
    dm.setup("fit")
    num_classes = dm.config.data.num_classes

    model = build_videomae_ordinal(config["model"], num_classes).to(DEVICE).eval()
    state = torch.load(_best_ckpt(run), map_location=DEVICE)["state_dict"]
    # Strip the LightningModule "model." prefix.
    model.load_state_dict({k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")})

    preds, targets = [], []
    for batch in dm.val_dataloader():
        clip = batch["AP"].to(DEVICE)
        logits = model(clip)
        preds.append(corn_class_probs(logits).argmax(1).cpu().numpy())
        targets.append(batch["label"].view(-1).numpy())
    return np.concatenate(preds), np.concatenate(targets)


@torch.no_grad()
def _predict_cvfsnet(run: str) -> Tuple[np.ndarray, np.ndarray]:
    from amticis_pipeline.dataset_loader import AmTICISDataModule
    from amticis_training.config import build_pipeline_config
    from amticis_training.lightning_module import AmTICISLightningModule
    from amticis_training.losses import build_loss
    from amticis_training.models import build_model

    config = _resolved_config(run)
    pipeline_config = build_pipeline_config(config)
    dm = AmTICISDataModule(config=pipeline_config)
    dm.setup("fit")
    num_classes = pipeline_config.data.num_classes

    model = build_model(config["model"], pipeline_config)
    criterion = build_loss(config["loss"], num_classes)
    module = AmTICISLightningModule(
        model=model, criterion=criterion, config=config, num_classes=num_classes
    )
    state = torch.load(_best_ckpt(run), map_location=DEVICE)["state_dict"]
    module.load_state_dict(state, strict=False)
    module = module.to(DEVICE).eval()

    preds, targets = [], []
    for batch in dm.val_dataloader():
        model_input = module._batch_to_model_input(batch)
        outputs = module._unwrap_outputs(module(model_input))
        fuse_logits = outputs[0]  # index 0 == "fuse"
        preds.append(fuse_logits.argmax(1).cpu().numpy())
        targets.append(batch["label"].view(-1).numpy())
    return np.concatenate(preds), np.concatenate(targets)


def _plot(results: dict) -> Path:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, (title, (preds, targets)) in zip(axes, results.items()):
        cm = confusion_matrix(targets, preds, labels=list(range(len(CLASS_NAMES))))
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / np.clip(row_sum, 1, None)
        acc = float((preds == targets).mean())
        qwk = cohen_kappa_score(
            targets, preds, labels=list(range(len(CLASS_NAMES))), weights="quadratic"
        )
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{title}\nacc={acc:.3f}  QWK={qwk:.3f}", fontsize=11)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(
                    j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion matrices (val, best epoch) — count / row-normalized", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = REPO_ROOT / "plots" / "confusion_matrices.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    return out


def main() -> None:
    results = {}
    for title, run in RUNS.items():
        try:
            if run.startswith("cvfsnet"):
                preds, targets = _predict_cvfsnet(run)
            else:
                preds, targets = _predict_videomae(run)
            results[title] = (preds, targets)
            print(f"{title}: n={len(targets)} acc={(preds==targets).mean():.3f}")
        except Exception as error:  # keep going so partial plots still render
            print(f"[WARN] {title} failed: {type(error).__name__}: {error}")
    if results:
        _plot(results)


if __name__ == "__main__":
    main()
