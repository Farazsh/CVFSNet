"""Compare VideoMAE vs CVFSNet binary confusion matrices, view by view.

Reads the per-run metrics CSVs exported by ``plot_confusion_binary_runs.py``
(CVFSNet) and ``mae_ordinal/plot_confusion_binary_runs.py`` (VideoMAE), matches
runs by view (AP<->Coronal, Sagittal<->Sagittal, Dual<->Fusion), and produces
one figure per view with two side-by-side panels (VideoMAE | CVFSNet). Reuses
the same ``_plot_panel`` function as ``plot_confusion_binary_runs.py`` so each
panel's style/annotations match exactly.

Usage:
    uv run python -m amticis_training.plot_confusion_comparison_videomae_cvfsnet
    uv run python -m amticis_training.plot_confusion_comparison_videomae_cvfsnet \
        --cvfsnet-csv Assets/confusion_matrix_metrics_cvfsnet_binary_runs_mod3.csv \
        --videomae-csv Assets/confusion_matrix_metrics_videomae_binary_runs.csv

Outputs (one per view):
    plots/confusion_matrix_comparison_ap_coronal.png
    plots/confusion_matrix_comparison_sagittal.png
    plots/confusion_matrix_comparison_fusion_dual.png
    plots/confusion_matrix_comparison_ap_vs_fusion.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from amticis_training.plot_confusion_binary_runs import _plot_panel
from amticis_training.plot_confusion_from_csv import _reconstruct_preds_targets

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CVFSNET_CSV = REPO_ROOT / "Assets" / "confusion_matrix_metrics_cvfsnet_binary_runs_mod3.csv"
DEFAULT_VIDEOMAE_CSV = REPO_ROOT / "Assets" / "confusion_matrix_metrics_videomae_binary_runs.csv"

# (output slug, view label, cvfsnet title, videomae title)
VIEW_MATCHES = [
    ("ap_coronal", "AP / Coronal", "Coronal", "AP"),
    ("sagittal", "Sagittal", "Sagittal", "Sagittal"),
    ("fusion_dual", "Fusion / Dual", "Fusion", "Dual"),
    # Cross-view comparison: VideoMAE's best single-view model vs CVFSNet's
    # best multi-view (fused) model.
    ("ap_vs_fusion", "AP vs Fusion", "Fusion", "AP"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cvfsnet-csv", type=Path, default=DEFAULT_CVFSNET_CSV)
    parser.add_argument("--videomae-csv", type=Path, default=DEFAULT_VIDEOMAE_CSV)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_rows_by_title(csv_path: Path) -> Dict[str, dict]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        return {row["title"]: row for row in csv.DictReader(handle)}


def _metrics_from_row(row: dict) -> dict:
    tp, tn, fp, fn = int(row["tp"]), int(row["tn"]), int(row["fp"]), int(row["fn"])
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": int(row.get("n", tp + tn + fp + fn)),
        "accuracy": float(row["accuracy"]),
        "f1_macro": float(row["f1_macro"]),
        "auroc": float(row["auroc"]),
        "auprc": float(row["auprc"]),
    }


def _plot_view(
    ax_pair,
    view_label: str,
    videomae_title: str,
    videomae_metrics: dict,
    cvfsnet_title: str,
    cvfsnet_metrics: dict,
) -> None:
    mae_preds, mae_targets = _reconstruct_preds_targets(
        videomae_metrics["tp"], videomae_metrics["tn"], videomae_metrics["fp"], videomae_metrics["fn"]
    )
    _plot_panel(
        ax_pair[0],
        f"VideoMAE {videomae_title} (n={videomae_metrics['n']})",
        mae_preds,
        mae_targets,
        videomae_metrics,
    )

    cvf_preds, cvf_targets = _reconstruct_preds_targets(
        cvfsnet_metrics["tp"], cvfsnet_metrics["tn"], cvfsnet_metrics["fp"], cvfsnet_metrics["fn"]
    )
    _plot_panel(
        ax_pair[1],
        f"CVFSNet {cvfsnet_title} (n={cvfsnet_metrics['n']})",
        cvf_preds,
        cvf_targets,
        cvfsnet_metrics,
    )


def main() -> None:
    args = parse_args()
    cvfsnet_csv = _resolve(args.cvfsnet_csv)
    videomae_csv = _resolve(args.videomae_csv)

    cvfsnet_rows = _load_rows_by_title(cvfsnet_csv)
    videomae_rows = _load_rows_by_title(videomae_csv)

    for slug, view_label, cvfsnet_title, videomae_title in VIEW_MATCHES:
        mae_metrics = _metrics_from_row(videomae_rows[videomae_title])
        cvf_metrics = _metrics_from_row(cvfsnet_rows[cvfsnet_title])

        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4))
        _plot_view(axes, view_label, videomae_title, mae_metrics, cvfsnet_title, cvf_metrics)

        fig.suptitle(f"VideoMAE vs CVFSNet \u2014 {view_label} view", fontsize=13)
        fig.tight_layout(rect=[0, 0.06, 1, 0.9])
        out = REPO_ROOT / "plots" / f"confusion_matrix_comparison_{slug}.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(
            f"{view_label}: VideoMAE acc={mae_metrics['accuracy']:.3f} "
            f"auroc={mae_metrics['auroc']:.3f}  |  CVFSNet acc={cvf_metrics['accuracy']:.3f} "
            f"auroc={cvf_metrics['auroc']:.3f}  ->  {out}"
        )


if __name__ == "__main__":
    main()
