"""Evaluate best binary checkpoints and export val-set sigmoid scores to CSV.

    uv run python -m mae_ordinal.eval_binary
    uv run python -m mae_ordinal.eval_binary --run mae_binary_t012a_ap
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BINARY_RUNS = [
    "mae_binary_t012a_ap",
    "mae_binary_t012a_sag",
    "mae_binary_t012a_dual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run name under output_runs_lightning (repeatable). Default: all binary runs.",
    )
    return parser.parse_args()


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


def _load_model(config: dict):
    from .model import build_videomae_binary

    return build_videomae_binary(config["model"]).to(DEVICE).eval()


def _model_input(batch: dict, config: dict):
    input_cfg = config.get("model", {}).get("input", {})
    if input_cfg.get("type") == "dual_view_list":
        return [
            batch["AP"].to(DEVICE),
            batch["sagittal"].to(DEVICE),
        ]
    view = input_cfg.get("view", "AP")
    return batch[view].to(DEVICE)


@torch.no_grad()
def evaluate_run(run_name: str) -> Tuple[dict, Path]:
    config = _resolved_config(run_name)
    from .dataloader import build_datamodule

    datamodule = build_datamodule(
        pipeline_overrides=config["data"].get("pipeline_overrides"),
        config_path=config["data"].get("config_path", "amticis_pipeline/config.yaml"),
        project_root=config["data"].get("project_root", "."),
        preset=config["data"].get("preset"),
    )
    datamodule.setup("fit")

    model = _load_model(config)
    state = torch.load(_best_ckpt(run_name), map_location=DEVICE)["state_dict"]
    model.load_state_dict(
        {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    )

    rows: List[dict] = []
    for batch in datamodule.val_dataloader():
        model_input = _model_input(batch, config)
        logit = model(model_input).view(-1)
        scores = torch.sigmoid(logit).cpu().numpy()
        labels = batch["label"].view(-1).numpy()
        names = batch["name"]
        if isinstance(names, str):
            names = [names]
        for name, true_label, score in zip(names, labels, scores):
            pred = int(score >= 0.5)
            rows.append(
                {
                    "name": name,
                    "true_label": int(true_label),
                    "sigmoid_score": float(score),
                    "pred_label": pred,
                }
            )

    out_path = _run_dir(run_name) / "val_sigmoid_scores.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["name", "true_label", "sigmoid_score", "pred_label"]
        )
        writer.writeheader()
        writer.writerows(rows)

    y_true = [row["true_label"] for row in rows]
    y_pred = [row["pred_label"] for row in rows]
    y_score = [row["sigmoid_score"] for row in rows]
    summary = {
        "run": run_name,
        "n": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "auroc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.0,
        "csv": str(out_path),
    }
    return summary, out_path


def main() -> None:
    args = parse_args()
    runs: Iterable[str] = args.run or BINARY_RUNS
    summaries = []
    for run_name in runs:
        try:
            summary, out_path = evaluate_run(run_name)
            summaries.append(summary)
            print(
                f"{run_name}: n={summary['n']} acc={summary['accuracy']:.3f} "
                f"f1={summary['f1_macro']:.3f} auroc={summary['auroc']:.3f} -> {out_path}"
            )
        except Exception as error:
            print(f"[WARN] {run_name} failed: {type(error).__name__}: {error}")

    if summaries:
        results_path = REPO_ROOT / "mae_ordinal" / "docs" / "BINARY_RESULTS_REPORT.md"
        _append_results(results_path, summaries)


def _append_results(path: Path, summaries: List[dict]) -> None:
    lines = ["\n## Eval summary (best checkpoint, val set)\n"]
    lines.append("| Run | N | Accuracy | F1 macro | AUROC | CSV |")
    lines.append("|-----|--:|---------:|---------:|------:|-----|")
    for row in summaries:
        lines.append(
            f"| {row['run']} | {row['n']} | {row['accuracy']:.3f} | "
            f"{row['f1_macro']:.3f} | {row['auroc']:.3f} | `{row['csv']}` |"
        )
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "## Eval summary (best checkpoint, val set)"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip()
    path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"updated {path}")


if __name__ == "__main__":
    main()
