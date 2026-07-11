"""Plot binary confusion matrices for VideoMAE runs vs collapsed CVFSNet baselines.

    uv run python -m mae_ordinal.plot_confusion_binary

Outputs: plots/confusion_matrices_binary.png
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["T012a", "T2b3"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COMPARISONS = [
    ("VideoMAE AP", "mae_binary_t012a_ap", "CVFSNet AP (collapsed)", "cvfsnet_pt_fuse01_cor_v1"),
    ("VideoMAE sagittal", "mae_binary_t012a_sag", "CVFSNet sag (collapsed)", "cvfsnet_pt_fuse01_sag_v1"),
    ("VideoMAE dual", "mae_binary_t012a_dual", "CVFSNet dual (collapsed)", "cvfsnet_pt_fuse01_fusion_v1"),
]


def _run_dir(name: str) -> Path:
    return REPO_ROOT / "output_runs_lightning" / name


def _best_ckpt(name: str) -> Path:
    ckpts = sorted((_run_dir(name) / "csv").glob("version_*/checkpoints/epoch=*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint for {name}")
    return ckpts[-1]


def _resolved_config(name: str) -> dict:
    with open(_run_dir(name) / "resolved_training_config.yaml") as handle:
        return yaml.safe_load(handle)


def collapse_fuse01_to_binary(preds: np.ndarray) -> np.ndarray:
    """Map fuse01 classes 0,1 -> 0 (T012a) and 2,3 -> 1 (T2b3)."""
    return np.where(preds <= 1, 0, 1)


def collapse_targets_fuse01_to_binary(targets: np.ndarray) -> np.ndarray:
    return collapse_fuse01_to_binary(targets)


@torch.no_grad()
def _predict_videomae_binary(run: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from .dataloader import build_datamodule
    from .model import build_videomae_binary

    config = _resolved_config(run)
    datamodule = build_datamodule(
        pipeline_overrides=config["data"].get("pipeline_overrides"),
        config_path=config["data"].get("config_path", "amticis_pipeline/config.yaml"),
        project_root=config["data"].get("project_root", "."),
        preset=config["data"].get("preset"),
    )
    datamodule.setup("fit")

    model = build_videomae_binary(config["model"]).to(DEVICE).eval()
    state = torch.load(_best_ckpt(run), map_location=DEVICE)["state_dict"]
    model.load_state_dict(
        {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    )

    input_cfg = config.get("model", {}).get("input", {})
    dual = input_cfg.get("type") == "dual_view_list"
    view = input_cfg.get("view", "AP")

    preds, targets, scores = [], [], []
    for batch in datamodule.val_dataloader():
        if dual:
            model_input = [batch["AP"].to(DEVICE), batch["sagittal"].to(DEVICE)]
        else:
            model_input = batch[view].to(DEVICE)
        logit = model(model_input).view(-1)
        score = torch.sigmoid(logit).cpu().numpy()
        pred = (score >= 0.5).astype(np.int64)
        preds.append(pred)
        scores.append(score)
        targets.append(batch["label"].view(-1).numpy())
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(scores)


@torch.no_grad()
def _predict_cvfsnet_collapsed(run: str) -> Tuple[np.ndarray, np.ndarray]:
    from amticis_pipeline.dataset_loader import AmTICISDataModule
    from amticis_training.config import build_pipeline_config
    from amticis_training.lightning_module import AmTICISLightningModule
    from amticis_training.losses import build_loss
    from amticis_training.models import build_model

    config = _resolved_config(run)
    pipeline_config = build_pipeline_config(config)
    datamodule = AmTICISDataModule(config=pipeline_config)
    datamodule.setup("fit")
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
    for batch in datamodule.val_dataloader():
        model_input = module._batch_to_model_input(batch)
        outputs = module._unwrap_outputs(module(model_input))
        fuse_preds = outputs[0].argmax(1).cpu().numpy()
        preds.append(collapse_fuse01_to_binary(fuse_preds))
        targets.append(collapse_targets_fuse01_to_binary(batch["label"].view(-1).numpy()))
    return np.concatenate(preds), np.concatenate(targets)


def _plot_panel(ax, title: str, preds: np.ndarray, targets: np.ndarray) -> None:
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.clip(row_sum, 1, None)
    acc = float(accuracy_score(targets, preds))
    f1 = float(f1_score(targets, preds, average="macro"))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"{title}\nacc={acc:.3f}  f1={f1:.3f}", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=9,
            )
    fig = ax.figure
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    n_rows = len(COMPARISONS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10.5, 4.2 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for row_idx, (mae_title, mae_run, base_title, base_run) in enumerate(COMPARISONS):
        for col_idx, (title, run, predictor) in enumerate(
            [
                (mae_title, mae_run, "videomae"),
                (base_title, base_run, "cvfsnet"),
            ]
        ):
            ax = axes[row_idx, col_idx]
            try:
                if predictor == "videomae":
                    preds, targets, _ = _predict_videomae_binary(run)
                else:
                    preds, targets = _predict_cvfsnet_collapsed(run)
                _plot_panel(ax, title, preds, targets)
                print(f"{title}: n={len(targets)} acc={(preds==targets).mean():.3f}")
            except Exception as error:
                ax.set_title(f"{title}\n(unavailable)")
                ax.axis("off")
                print(f"[WARN] {title} failed: {type(error).__name__}: {error}")

    fig.suptitle(
        "Binary T012a vs T2b3 confusion matrices (val, best epoch)\n"
        "CVFSNet baselines: fuse01 predictions collapsed to binary",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = REPO_ROOT / "plots" / "confusion_matrices_binary.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
