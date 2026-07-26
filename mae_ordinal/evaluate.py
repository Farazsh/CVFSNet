"""Reload a binary VideoMAE checkpoint, evaluate it, and export artifacts.

Artifacts match the dinov3 evaluation schema: ``val_scores.csv``,
``evaluation.json``, and a dated Excel workbook.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from amticis_pipeline.utils import parse_label_token
from dinov3.evaluate import EXCEL_COLUMNS
from dinov3.metrics import binary_metrics, tune_macro_f1_threshold

from .dataloader import build_datamodule
from .model import build_videomae_binary

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    trainer_cfg = config.get("trainer", {})
    if trainer_cfg.get("accelerator") == "gpu" and torch.cuda.is_available():
        devices = trainer_cfg.get("devices", [0])
        index = int(devices[0] if isinstance(devices, list) else 0)
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def build_binary_model(config: Mapping[str, Any]) -> torch.nn.Module:
    return build_videomae_binary(config["model"])


def model_input_from_batch(batch: Mapping[str, Any], config: Mapping[str, Any], device: torch.device):
    input_cfg = config.get("model", {}).get("input", {})
    if input_cfg.get("type") == "dual_view_list":
        return [
            batch["AP"].to(device, non_blocking=True).contiguous(),
            batch["sagittal"].to(device, non_blocking=True).contiguous(),
        ]
    view = input_cfg.get("view", "AP")
    return batch[view].to(device, non_blocking=True).contiguous()


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    dataloader,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = model.to(device).eval()
    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    names: list[str] = []
    grades: list[str] = []
    for batch in dataloader:
        logits = model(model_input_from_batch(batch, config, device)).view(-1)
        logits_all.append(logits.cpu())
        targets_all.append(batch["label"].view(-1).cpu())
        batch_names = batch["name"]
        if isinstance(batch_names, str):
            batch_names = [batch_names]
        for name in batch_names:
            names.append(str(name))
            grades.append(parse_label_token(str(name)))
    if not logits_all:
        raise RuntimeError("Validation dataloader produced no batches.")
    logits = torch.cat(logits_all).numpy()
    targets = torch.cat(targets_all).numpy().astype(np.int64)
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    return {
        "names": names,
        "raw_grades": grades,
        "logits": logits,
        "targets": targets,
        "probabilities": probabilities,
    }


def checkpoint_epoch(checkpoint_path: str | Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", 0))


def load_model_from_checkpoint(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> torch.nn.Module:
    device = device or resolve_device(config)
    model = build_binary_model(config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(
        {key[len("model.") :]: value for key, value in state.items() if key.startswith("model.")}
    )
    return model.to(device).eval()


def write_outputs(
    run_dir: Path,
    config: Mapping[str, Any],
    checkpoint_path: Path,
    predictions: Mapping[str, Any],
    threshold: float,
    macro_f1: float,
    metrics: Mapping[str, float],
    epoch: int,
    train_size: int = 261,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_cfg = config.get("evaluation", {})
    binary_predictions = (predictions["probabilities"] >= threshold).astype(np.int64)
    scores = pd.DataFrame(
        {
            "study_name": predictions["names"],
            "raw_grade": predictions["raw_grades"],
            "target": predictions["targets"],
            "logit": predictions["logits"],
            "probability": predictions["probabilities"],
            "prediction": binary_predictions,
        }
    )
    scores_path = run_dir / str(evaluation_cfg.get("scores_filename", "val_scores.csv"))
    scores.to_csv(scores_path, index=False)

    model_cfg = config.get("model", {})
    payload = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "best_epoch": int(epoch),
        "threshold": float(threshold),
        "threshold_strategy": str(
            config.get("threshold", {}).get("strategy", "maximize_macro_f1")
        ),
        "threshold_macro_f1": float(macro_f1),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "seed": int(config.get("seed", 0)),
        "model_name": str(model_cfg.get("name", "videomae_binary")),
        "pretrained_name": str(
            model_cfg.get("params", {}).get("pretrained_name", "")
        ),
        "train_size": int(train_size),
        "validation_size": int(len(scores)),
        "selection_and_reporting_split": "val",
        "bias_warning": (
            "The validation cohort selected the checkpoint and threshold and is also "
            "the reported evaluation cohort; metrics are not held-out estimates."
        ),
    }
    json_path = run_dir / str(evaluation_cfg.get("json_filename", "evaluation.json"))
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    metadata = dict(evaluation_cfg.get("excel_metadata", {}))
    metadata.setdefault("Model", str(model_cfg.get("name", "videomae_binary")))
    metadata.setdefault("Dimension", "3D")
    metadata.setdefault("Label1", "BinaryTICI")
    metadata.setdefault("Label2", "T012a_vs_T2b3")
    metadata.setdefault("InputSize", "16x224x224")
    metadata.setdefault("Optimizer", "AdamW")
    metadata.setdefault("Loop", "train_val")
    metadata.setdefault("ExperimentVersion", str(config.get("seed", "")))
    metadata.setdefault("ValidationSplit", "val")
    row = {
        "Date": datetime.now().strftime("%Y%m%d"),
        **metadata,
        "Epoch": int(epoch),
        "Auroc": float(metrics["auroc"]),
        "Auprc": float(metrics["auprc"]),
        "Accuracy": float(metrics["accuracy"]),
        "Precision": float(metrics["precision"]),
        "Recall": float(metrics["recall"]),
        "F1": float(metrics["f1"]),
        "Specificity": float(metrics["specificity"]),
    }
    workbook_suffix = str(
        evaluation_cfg.get("workbook_suffix", f"{config['run']['name']}_eval.xlsx")
    )
    workbook_path = run_dir / f"{row['Date']}_{workbook_suffix}"
    pd.DataFrame([[row[column] for column in EXCEL_COLUMNS]], columns=EXCEL_COLUMNS).to_excel(
        workbook_path,
        index=False,
        engine="xlsxwriter",
    )
    return {"scores": scores_path, "json": json_path, "workbook": workbook_path}


def evaluate_checkpoint(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    data_cfg = config["data"]
    datamodule = build_datamodule(
        pipeline_overrides=data_cfg.get("pipeline_overrides"),
        config_path=data_cfg.get("config_path", "amticis_pipeline/config.yaml"),
        project_root=data_cfg.get("project_root", "."),
        preset=data_cfg.get("preset"),
    )
    datamodule.setup("fit")
    device = resolve_device(config)
    model = load_model_from_checkpoint(config, checkpoint_path, device=device)
    predictions = collect_predictions(model, datamodule.val_dataloader(), config, device)
    threshold, macro_f1 = tune_macro_f1_threshold(
        predictions["probabilities"], predictions["targets"]
    )
    metrics = binary_metrics(predictions["probabilities"], predictions["targets"], threshold)
    epoch = checkpoint_epoch(checkpoint_path)
    train_ds = getattr(datamodule, "_datasets", {}).get("train")
    train_size = len(train_ds) if train_ds is not None else 261
    paths = write_outputs(
        Path(run_dir),
        config,
        Path(checkpoint_path),
        predictions,
        threshold,
        macro_f1,
        metrics,
        epoch,
        train_size=train_size,
    )
    return {"threshold": threshold, "metrics": metrics, "paths": paths}


def find_best_checkpoint(run_dir: Path) -> Path:
    """Locate the best checkpoint under a run directory."""
    candidates = []
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        candidates.extend(checkpoints_dir.glob("*.ckpt"))
    candidates.extend(sorted((run_dir / "csv").glob("version_*/checkpoints/*.ckpt")))
    # Prefer non-last checkpoints that encode val_auroc in the filename.
    ranked = sorted(
        candidates,
        key=lambda path: (
            0 if "val_auroc" in path.name else 1,
            0 if path.name != "last.ckpt" else 1,
            -path.stat().st_mtime,
        ),
    )
    if not ranked:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")
    return ranked[0]


def load_resolved_config(run_dir: Path) -> dict[str, Any]:
    for name in ("resolved_training_config.yaml", "resolved_config.yaml"):
        path = run_dir / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
    raise FileNotFoundError(f"No resolved config in {run_dir}")
