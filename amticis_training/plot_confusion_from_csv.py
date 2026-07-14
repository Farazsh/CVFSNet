"""Plot binary confusion matrices from a previously exported (or hand-edited)
metrics CSV, reusing the exact same ``_plot_panel`` panel-drawing function as
``amticis_training/plot_confusion_binary_runs.py`` so the figure style
matches. Useful for re-plotting a CSV that was manually modified (e.g. to
explore what-if metrics) without re-running inference.

The CSV must contain, at minimum, the columns written by
``plot_confusion_binary_runs.py`` / ``mae_ordinal/plot_confusion_binary_runs.py``:
``title``, ``n``, ``tp``, ``tn``, ``fp``, ``fn``, ``accuracy``, ``f1_macro``,
``auroc``, ``auprc``.

Usage:
    uv run python -m amticis_training.plot_confusion_from_csv \
        Assets/confusion_matrix_metrics_cvfsnet_binary_runs_mod.csv

Output: plots/<csv-stem, with 'confusion_matrix_metrics_' -> 'confusion_matrices_'>.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from amticis_training.plot_confusion_binary_runs import _plot_panel

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Metrics CSV with tp/tn/fp/fn and accuracy/f1_macro/auroc/auprc columns.",
    )
    parser.add_argument(
        "--title-prefix",
        default="",
        help="Optional prefix for each panel title (e.g. 'CVFSNet').",
    )
    return parser.parse_args()


def _reconstruct_preds_targets(tp: int, tn: int, fp: int, fn: int):
    """Synthesize target/pred label arrays that reproduce the given 2x2 counts.

    ``_plot_panel`` recomputes the confusion matrix from raw preds/targets via
    ``sklearn.metrics.confusion_matrix``, so this recovers the exact same
    matrix without needing to change that function.
    """
    targets = np.array([0] * tn + [0] * fp + [1] * fn + [1] * tp)
    preds = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)
    return preds, targets


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path if args.csv_path.is_absolute() else REPO_ROOT / args.csv_path
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    fig, axes = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 5.4))
    if len(rows) == 1:
        axes = [axes]

    prefix = f"{args.title_prefix} " if args.title_prefix else ""
    for ax, row in zip(axes, rows):
        tp, tn, fp, fn = int(row["tp"]), int(row["tn"]), int(row["fp"]), int(row["fn"])
        preds, targets = _reconstruct_preds_targets(tp, tn, fp, fn)
        metrics = {
            "n": int(row.get("n", tp + tn + fp + fn)),
            "accuracy": float(row["accuracy"]),
            "f1_macro": float(row["f1_macro"]),
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
        }
        title = f"{prefix}{row['title']} (n={metrics['n']})"
        _plot_panel(ax, title, preds, targets, metrics)
        print(
            f"{row['title']}: n={metrics['n']} acc={metrics['accuracy']:.3f} "
            f"f1={metrics['f1_macro']:.3f} auroc={metrics['auroc']:.3f} "
            f"auprc={metrics['auprc']:.3f}"
        )

    fig.suptitle(f"Binary confusion matrices)", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.9])
    plot_name = csv_path.stem.replace("confusion_matrix_metrics_", "confusion_matrices_") + ".png"
    out = REPO_ROOT / "plots" / plot_name
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
