"""Reference-compatible binary metrics and deterministic threshold tuning."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _arrays(
    probabilities: Sequence[float] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    truth = np.asarray(targets, dtype=np.int64).reshape(-1)
    if probs.size == 0 or probs.shape != truth.shape:
        raise ValueError("Probabilities and targets must be non-empty and equally sized.")
    if not np.isfinite(probs).all():
        raise ValueError("Probabilities contain NaN or infinity.")
    if not np.isin(truth, (0, 1)).all() or np.unique(truth).size != 2:
        raise ValueError("Binary metrics require both target classes 0 and 1.")
    return probs, truth


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(
    probabilities: Sequence[float] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute the seven metrics from ``train_and_test.py`` plus counts."""
    probs, truth = _arrays(probabilities, targets)
    predictions = (probs >= float(threshold)).astype(np.int64)
    tp = int(np.logical_and(truth == 1, predictions == 1).sum())
    tn = int(np.logical_and(truth == 0, predictions == 0).sum())
    fp = int(np.logical_and(truth == 0, predictions == 1).sum())
    fn = int(np.logical_and(truth == 1, predictions == 0).sum())
    accuracy = _divide(tp + tn, tp + tn + fp + fn)
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    f1 = _divide(2.0 * precision * recall, precision + recall)
    return {
        "auroc": float(roc_auc_score(truth, probs)),
        "auprc": float(average_precision_score(truth, probs)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _macro_f1(probs: np.ndarray, truth: np.ndarray, threshold: float) -> float:
    pred = (probs >= threshold).astype(np.int64)
    scores = []
    for class_id in (0, 1):
        tp = int(np.logical_and(truth == class_id, pred == class_id).sum())
        fp = int(np.logical_and(truth != class_id, pred == class_id).sum())
        fn = int(np.logical_and(truth == class_id, pred != class_id).sum())
        precision = _divide(tp, tp + fp)
        recall = _divide(tp, tp + fn)
        scores.append(_divide(2.0 * precision * recall, precision + recall))
    return float(np.mean(scores))


def tune_macro_f1_threshold(
    probabilities: Sequence[float] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
) -> tuple[float, float]:
    """Maximize macro F1; ties select the threshold closest to 0.5."""
    probs, truth = _arrays(probabilities, targets)
    candidates = np.unique(np.concatenate((probs, np.array([0.0, 0.5, 1.0]))))
    scored = [(float(value), _macro_f1(probs, truth, float(value))) for value in candidates]
    best_score = max(score for _, score in scored)
    tied = [threshold for threshold, score in scored if np.isclose(score, best_score)]
    threshold = min(tied, key=lambda value: (abs(value - 0.5), value))
    return float(threshold), float(best_score)
