"""Plot side-by-side binary confusion matrices for the three VideoMAE runs
(AP, sagittal, dual) with accuracy / F1 / AUROC / AUPRC annotated per panel.

Reads the val-set sigmoid score CSVs already exported by ``eval_binary``:
    output_runs_lightning/mae_binary_t012a_ap/val_sigmoid_scores.csv
    output_runs_lightning/mae_binary_t012a_sag/val_sigmoid_scores.csv
    output_runs_lightning/mae_binary_t012a_dual/val_sigmoid_scores.csv

Usage:
    uv run python -m mae_ordinal.plot_confusion_binary_runs

Outputs:
    plots/confusion_matrices_binary_runs.png
    Assets/confusion_matrix_metrics_videomae_binary_runs.csv
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["T012a", "T2b3"]

RUNS = [
    ("AP", "mae_binary_t012a_ap"),
    ("Sagittal", "mae_binary_t012a_sag"),
    ("Dual", "mae_binary_t012a_dual"),
]

METRICS_CSV_FIELDS = [
    "experiment",
    "run_name",
    "title",
    "n",
    "class_0_name",
    "class_1_name",
    "positive_class",
    "tp",
    "tn",
    "fp",
    "fn",
    "accuracy",
    "f1_macro",
    "auroc",
    "auprc",
    "checkpoint_path",
    "config_path",
    "scores_source",
    "plot_path",
    "generated_at",
]


def _run_dir(name: str) -> Path:
    return REPO_ROOT / "output_runs_lightning" / name


def _best_ckpt(name: str) -> Path:
    ckpts = sorted((_run_dir(name) / "csv").glob("version_*/checkpoints/epoch=*.ckpt"))
    return ckpts[-1] if ckpts else Path("")


def _scores_csv(run_name: str) -> Path:
    return _run_dir(run_name) / "val_sigmoid_scores.csv"


def _load_scores(run_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = _scores_csv(run_name)
    targets: List[int] = []
    scores: List[float] = []
    preds: List[int] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            targets.append(int(row["true_label"]))
            scores.append(float(row["sigmoid_score"]))
            preds.append(int(row["pred_label"]))
    return np.array(preds), np.array(targets), np.array(scores)


def _compute_metrics(preds: np.ndarray, targets: np.ndarray, scores: np.ndarray) -> dict:
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    return {
        "n": len(targets),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy_score(targets, preds)),
        "f1_macro": float(f1_score(targets, preds, average="macro")),
        "auroc": float(roc_auc_score(targets, scores)),
        "auprc": float(average_precision_score(targets, scores)),
    }


def _write_metrics_csv(rows: List[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}")


def _plot_panel(ax, title: str, preds: np.ndarray, targets: np.ndarray, metrics: dict) -> None:
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.clip(row_sum, 1, None)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=12, fontweight="bold")
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
                fontsize=10,
            )
    fig = ax.figure
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    caption = (
        f"Accuracy={metrics['accuracy']:.3f}   F1={metrics['f1_macro']:.3f}\n"
        f"AUROC={metrics['auroc']:.3f}   AUPRC={metrics['auprc']:.3f}"
    )
    ax.text(
        0.5,
        -0.32,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
    )


def main() -> None:
    fig, axes = plt.subplots(1, len(RUNS), figsize=(5.2 * len(RUNS), 5.4))
    generated_at = datetime.now(timezone.utc).isoformat()
    plot_path = REPO_ROOT / "plots" / "confusion_matrices_binary_runs.png"
    csv_rows: List[dict] = []

    for ax, (title, run_name) in zip(axes, RUNS):
        preds, targets, scores = _load_scores(run_name)
        metrics = _compute_metrics(preds, targets, scores)
        _plot_panel(ax, f"VideoMAE {title} (n={metrics['n']})", preds, targets, metrics)
        print(
            f"{title}: n={metrics['n']} acc={metrics['accuracy']:.3f} "
            f"f1={metrics['f1_macro']:.3f} auroc={metrics['auroc']:.3f} "
            f"auprc={metrics['auprc']:.3f}"
        )
        csv_rows.append(
            {
                "experiment": "videomae_binary",
                "run_name": run_name,
                "title": title,
                "n": metrics["n"],
                "class_0_name": CLASS_NAMES[0],
                "class_1_name": CLASS_NAMES[1],
                "positive_class": CLASS_NAMES[1],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "accuracy": f"{metrics['accuracy']:.6f}",
                "f1_macro": f"{metrics['f1_macro']:.6f}",
                "auroc": f"{metrics['auroc']:.6f}",
                "auprc": f"{metrics['auprc']:.6f}",
                "checkpoint_path": str(_best_ckpt(run_name)),
                "config_path": str(_run_dir(run_name) / "resolved_training_config.yaml"),
                "scores_source": str(_scores_csv(run_name)),
                "plot_path": str(plot_path),
                "generated_at": generated_at,
            }
        )

    fig.suptitle(
        "Binary T012a vs T2b3 confusion matrices (val, best checkpoint)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    plot_path.parent.mkdir(exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"saved {plot_path}")

    metrics_csv = REPO_ROOT / "Assets" / "confusion_matrix_metrics_videomae_binary_runs.csv"
    _write_metrics_csv(csv_rows, metrics_csv)


if __name__ == "__main__":
    main()
