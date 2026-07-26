"""Evaluate best binary checkpoints and export dinov3-style artifacts.

    uv run python -m mae_ordinal.eval_binary
    uv run python -m mae_ordinal.eval_binary --run mae_binary_t012a_ap
    uv run python -m mae_ordinal.eval_binary --run mae_v2_binary_t012a_ap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .evaluate import (
    REPO_ROOT,
    evaluate_checkpoint,
    find_best_checkpoint,
    load_resolved_config,
)

BINARY_RUNS = [
    "mae_binary_t012a_ap",
    "mae_binary_t012a_sag",
    "mae_binary_t012a_dual",
    "mae_v2_binary_t012a_ap",
    "mae_v2_binary_t012a_sag",
    "mae_v2_binary_t012a_dual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run name under output_runs_lightning (repeatable). Default: known binary runs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path (requires a single --run).",
    )
    return parser.parse_args()


def _run_dir(name: str) -> Path:
    return REPO_ROOT / "output_runs_lightning" / name


def evaluate_run(run_name: str, checkpoint: Path | None = None) -> dict:
    run_dir = _run_dir(run_name)
    config = load_resolved_config(run_dir)
    ckpt = checkpoint or find_best_checkpoint(run_dir)
    result = evaluate_checkpoint(config, ckpt, run_dir)
    metrics = result["metrics"]
    n = len(pd.read_csv(result["paths"]["scores"]))
    return {
        "run": run_name,
        "n": n,
        "accuracy": float(metrics["accuracy"]),
        "f1": float(metrics["f1"]),
        "auroc": float(metrics["auroc"]),
        "threshold": float(result["threshold"]),
        "scores": str(result["paths"]["scores"]),
        "json": str(result["paths"]["json"]),
        "workbook": str(result["paths"]["workbook"]),
    }


def main() -> None:
    args = parse_args()
    runs: Iterable[str] = args.run or BINARY_RUNS
    if args.checkpoint is not None and len(list(runs)) != 1:
        raise SystemExit("--checkpoint requires exactly one --run")
    summaries: List[dict] = []
    for run_name in runs:
        try:
            summary = evaluate_run(run_name, checkpoint=args.checkpoint)
            summaries.append(summary)
            print(
                f"{run_name}: n={summary['n']} acc={summary['accuracy']:.3f} "
                f"f1={summary['f1']:.3f} auroc={summary['auroc']:.3f} "
                f"thr={summary['threshold']:.3f} -> {summary['scores']}"
            )
            print(json.dumps({"threshold": summary["threshold"], "metrics": {
                "accuracy": summary["accuracy"],
                "f1": summary["f1"],
                "auroc": summary["auroc"],
            }}, indent=2))
        except Exception as error:
            print(f"[WARN] {run_name} failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
