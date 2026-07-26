"""Reload a DINOv3 checkpoint, evaluate it, and export CSV/JSON/XLSX artifacts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv

from amticis_training.config import deep_update, parse_overrides

from .dataloader import DINOv3DataModule
from .lightning_module import DINOv3LightningModule
from .losses import build_loss
from .metrics import binary_metrics, tune_macro_f1_threshold
from .model import DINOv3BinaryClassifier

EXCEL_COLUMNS = [
    "Date",
    "Model",
    "Dimension",
    "Label1",
    "Label2",
    "InputSize",
    "Optimizer",
    "Loop",
    "ExperimentVersion",
    "ValidationSplit",
    "Epoch",
    "Auroc",
    "Auprc",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Specificity",
]


def load_config(path: str | Path, overrides=None) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    deep_update(config, overrides)
    config["_config_path"] = str(path.resolve())
    return config


def load_processor(config: Mapping[str, Any], token: str | None):
    model_cfg = config["model"]
    backend = str(model_cfg.get("backend", "huggingface")).lower()
    if backend == "lightly":
        normalization = model_cfg.get("lightly_normalization", {})
        mean = list(normalization.get("mean", ()))
        std = list(normalization.get("std", ()))
        if len(mean) != 3 or len(std) != 3 or any(float(value) <= 0 for value in std):
            raise ValueError("Lightly normalization requires three means and positive stds.")
        return None, mean, std
    if backend != "huggingface":
        raise ValueError("model.backend must be either 'huggingface' or 'lightly'.")
    from transformers import AutoImageProcessor

    try:
        processor = AutoImageProcessor.from_pretrained(
            model_cfg["pretrained_name"],
            token=token,
            revision=model_cfg.get("revision"),
        )
    except Exception as error:
        if "403" in str(error) or "gated" in str(error).lower():
            raise RuntimeError(
                "DINOv3 is gated. Accept the model license on Hugging Face and ensure "
                f"{model_cfg.get('token_env', 'HF_TOKEN')} belongs to that account."
            ) from error
        raise
    mean = list(getattr(processor, "image_mean", []))
    std = list(getattr(processor, "image_std", []))
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("The DINOv3 image processor must provide three-channel mean/std.")
    return processor, mean, std


def build_model(config: Mapping[str, Any], token: str | None) -> DINOv3BinaryClassifier:
    model_cfg = config["model"]
    return DINOv3BinaryClassifier(
        pretrained_name=str(model_cfg["pretrained_name"]),
        token=token,
        revision=model_cfg.get("revision"),
        dropout=float(model_cfg.get("dropout", 0.3)),
        expected_num_register_tokens=int(model_cfg.get("expected_num_register_tokens", 4)),
        backend=str(model_cfg.get("backend", "huggingface")),
        lightly_name=str(model_cfg.get("lightly_name", "dinov3/vits16")),
    )


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    trainer_cfg = config.get("trainer", {})
    if trainer_cfg.get("accelerator") == "gpu" and torch.cuda.is_available():
        devices = trainer_cfg.get("devices", [0])
        index = int(devices[0] if isinstance(devices, list) else 0)
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


@torch.inference_mode()
def collect_predictions(model, dataloader, device: torch.device) -> dict[str, Any]:
    model = model.to(device).eval()
    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    names: list[str] = []
    grades: list[str] = []
    for batch in dataloader:
        logits = model(batch["pixel_values"].to(device, non_blocking=True)).view(-1)
        logits_all.append(logits.cpu())
        targets_all.append(batch["label"].view(-1).cpu())
        names.extend(str(value) for value in batch["name"])
        grades.extend(str(value) for value in batch["raw_grade"])
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


def write_outputs(
    run_dir: Path,
    config: Mapping[str, Any],
    checkpoint_path: Path,
    predictions: Mapping[str, Any],
    threshold: float,
    macro_f1: float,
    metrics: Mapping[str, float],
    epoch: int,
    model_revision: str | None,
    train_size: int = 261,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_cfg = config["evaluation"]
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

    payload = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "best_epoch": int(epoch),
        "threshold": float(threshold),
        "threshold_strategy": "maximize_macro_f1",
        "threshold_macro_f1": float(macro_f1),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "seed": int(config["seed"]),
        "model_backend": str(config["model"].get("backend", "huggingface")),
        "model_name": str(
            config["model"].get("lightly_name")
            if config["model"].get("backend") == "lightly"
            else config["model"]["pretrained_name"]
        ),
        "model_revision": model_revision,
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

    metadata = dict(evaluation_cfg["excel_metadata"])
    backend = str(config["model"].get("backend", "huggingface"))
    metadata["Model"] = f"{backend}-dinov3-vits16"
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
    workbook_name = f"{row['Date']}_{evaluation_cfg['workbook_suffix']}"
    workbook_path = run_dir / workbook_name
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
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    token = os.environ.get(str(config["model"].get("token_env", "HF_TOKEN")))
    _, mean, std = load_processor(config, token)
    datamodule = DINOv3DataModule(config["data"], mean, std, project_root=project_root)
    datamodule.setup("validate")

    fresh_model = build_model(config, token)
    module = DINOv3LightningModule.load_from_checkpoint(
        str(checkpoint_path),
        model=fresh_model,
        loss_fn=build_loss(config["loss"]),
        config=config,
        map_location="cpu",
    )
    predictions = collect_predictions(module.model, datamodule.val_dataloader(), resolve_device(config))
    threshold, macro_f1 = tune_macro_f1_threshold(
        predictions["probabilities"], predictions["targets"]
    )
    metrics = binary_metrics(predictions["probabilities"], predictions["targets"], threshold)
    epoch = checkpoint_epoch(checkpoint_path)
    backbone_config = getattr(module.model.backbone, "config", None)
    revision = getattr(backbone_config, "_commit_hash", None)
    paths = write_outputs(
        Path(run_dir),
        config,
        Path(checkpoint_path),
        predictions,
        threshold,
        macro_f1,
        metrics,
        epoch,
        revision,
        train_size=len(datamodule.train_dataset),
    )
    return {"threshold": threshold, "metrics": metrics, "paths": paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, parse_overrides(args.set))
    run_cfg = config["run"]
    output_dir = args.output_dir or Path(run_cfg["output_dir"]) / run_cfg["name"]
    result = evaluate_checkpoint(config, args.checkpoint, output_dir)
    print(json.dumps({"threshold": result["threshold"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
