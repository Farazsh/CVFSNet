"""Shared utilities for corrected zero-shot TICI experiments."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from amticis_training.config import deep_update, parse_overrides

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("T012a", "T2b3")


def load_config(path: Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    deep_update(config, parse_overrides(overrides))
    return config


def load_dotenv() -> None:
    """Load repository credentials without making them part of run artifacts."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv as _load

        _load(env_path)
    except ImportError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scan_to_rgb_frames(
    clip: torch.Tensor,
    scaling: Mapping[str, Any],
) -> list[Image.Image]:
    """Convert one ``(1, T, H, W)`` scan to model-ready RGB frames."""
    if clip.ndim != 4 or clip.shape[0] != 1:
        raise ValueError(f"Expected scan shape (1, T, H, W), got {tuple(clip.shape)}.")
    arr = clip.detach().cpu().float().numpy()[0]
    if not np.isfinite(arr).all():
        raise FloatingPointError("DSA input contains NaN or infinity.")

    mode = str(scaling["mode"]).lower()
    if mode == "percentile":
        lower = float(scaling.get("lower", 1.0))
        upper = float(scaling.get("upper", 99.0))
        if not 0 <= lower < upper <= 100:
            raise ValueError("Percentile bounds must satisfy 0 <= lower < upper <= 100.")
        lo, hi = np.percentile(arr, [lower, upper])
    elif mode == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
    else:
        raise ValueError(f"Unknown intensity scaling mode '{mode}'.")
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError(f"Degenerate DSA intensity range: lo={lo}, hi={hi}.")

    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    frames = []
    for frame in scaled:
        gray = np.rint(frame * 255.0).astype(np.uint8)
        frames.append(Image.fromarray(np.repeat(gray[..., None], 3, axis=2), mode="RGB"))
    return frames


def assert_finite_scores(scores: torch.Tensor, study_name: str) -> torch.Tensor:
    scores = scores.detach().float().view(-1)
    if scores.numel() != 2:
        raise ValueError(f"Expected two class scores for {study_name}, got {scores.numel()}.")
    if not torch.isfinite(scores).all():
        raise FloatingPointError(
            f"Non-finite class score for {study_name}: {scores.cpu().tolist()}"
        )
    return scores


def score_row(
    *,
    study_name: str,
    true_label: int,
    scores: torch.Tensor,
    model_name: str,
    prompt_id: str,
    scaling_id: str,
    threshold: float,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scores = assert_finite_scores(scores, study_name)
    probabilities = torch.softmax(scores, dim=0)
    probability_1 = float(probabilities[1].item())
    row: dict[str, Any] = {
        "name": study_name,
        "true_label": int(true_label),
        "score_0": float(scores[0].item()),
        "score_1": float(scores[1].item()),
        "probability_t2b3": probability_1,
        "predicted_label": int(probability_1 >= threshold),
        "exact_tie": bool(scores[0].item() == scores[1].item()),
        "threshold": float(threshold),
        "model": model_name,
        "prompt_id": prompt_id,
        "scaling_id": scaling_id,
    }
    row.update(extra or {})
    return row


def make_internal_split(
    names: Sequence[str],
    labels: Sequence[int],
    *,
    tuning_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    if len(names) != len(labels):
        raise ValueError("Names and labels have different lengths.")
    development, tuning = train_test_split(
        list(names),
        test_size=tuning_fraction,
        random_state=seed,
        stratify=list(labels),
    )
    result = {"development": sorted(development), "tuning": sorted(tuning)}
    assert_disjoint_studies(result)
    return result


def patient_id(name: str) -> str:
    return Path(name).name.split("_", 1)[0]


def assert_disjoint_studies(splits: Mapping[str, Sequence[str]]) -> None:
    keys = list(splits)
    for index, left in enumerate(keys):
        left_ids = {patient_id(name) for name in splits[left]}
        for right in keys[index + 1 :]:
            overlap = left_ids & {patient_id(name) for name in splits[right]}
            if overlap:
                raise ValueError(f"Patient overlap between {left} and {right}: {sorted(overlap)}")


def _point_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    specificity = np.divide(
        np.array([cm[1, 1], cm[0, 0]], dtype=float),
        np.array([cm[1].sum(), cm[0].sum()], dtype=float),
        out=np.zeros(2, dtype=float),
        where=np.array([cm[1].sum(), cm[0].sum()]) != 0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "class_0_precision": float(precision[0]),
        "class_0_recall": float(recall[0]),
        "class_0_f1": float(f1[0]),
        "class_0_specificity": float(specificity[0]),
        "class_1_precision": float(precision[1]),
        "class_1_recall": float(recall[1]),
        "class_1_f1": float(f1[1]),
        "class_1_specificity": float(specificity[1]),
    }


def bootstrap_confidence_intervals(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    y_true = np.asarray([int(row["true_label"]) for row in rows])
    y_score = np.asarray([float(row["probability_t2b3"]) for row in rows])
    class_indices = [np.flatnonzero(y_true == class_id) for class_id in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("Bootstrap requires both classes.")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(replicates):
        draw = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        metrics = _point_metrics(y_true[draw], y_score[draw], threshold)
        for name, value in metrics.items():
            samples.setdefault(name, []).append(value)
    return {
        name: {
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
        }
        for name, values in samples.items()
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No predictions were produced.")
    y_true = np.asarray([int(row["true_label"]) for row in rows])
    y_score = np.asarray([float(row["probability_t2b3"]) for row in rows])
    if not np.isfinite(y_score).all():
        raise FloatingPointError("Prediction export contains non-finite probabilities.")
    metrics = _point_metrics(y_true, y_score, threshold)
    tie_rate = float(np.mean([bool(row["exact_tie"]) for row in rows]))
    return {
        "n": len(rows),
        "metrics": metrics,
        "tie_rate": tie_rate,
        "invalid_rate": 0.0,
        "bootstrap_95_ci": bootstrap_confidence_intervals(
            rows,
            threshold=threshold,
            replicates=bootstrap_replicates,
            seed=seed,
        ),
    }


def environment_metadata(model_revision: str) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "model_revision": model_revision,
        "git_commit": commit,
    }
    for package in ("transformers", "accelerate", "bitsandbytes"):
        try:
            module = __import__(package)
            metadata[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            metadata[package] = None
    if torch.cuda.is_available():
        metadata["gpus"] = [
            {
                "name": torch.cuda.get_device_name(index),
                "total_memory": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return metadata


def write_artifacts(
    *,
    config: Mapping[str, Any],
    split_name: str,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Path:
    run_cfg = config["run"]
    run_dir = REPO_ROOT / str(run_cfg["output_dir"]) / str(run_cfg["name"])
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
    with open(run_dir / "environment.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(metadata), handle, sort_keys=False)

    csv_path = run_dir / f"{split_name}_scores.csv"
    fieldnames = list(rows[0])
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(run_dir / f"{split_name}_metrics.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(summary), handle, sort_keys=False)
    _write_confusion_matrix(run_dir / f"{split_name}_confusion_matrix.png", rows)
    return csv_path


def write_split_manifest(path: Path, splits: Mapping[str, Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({key: list(value) for key, value in splits.items()}, handle, indent=2)


def _write_confusion_matrix(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    y_true = np.asarray([int(row["true_label"]) for row in rows])
    y_pred = np.asarray([int(row["predicted_label"]) for row in rows])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    normalized = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual")
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{cm[row, column]}\n{normalized[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if normalized[row, column] > 0.5 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
