"""Epoch-level classification metrics for AmTICIS training."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


class ClassificationMetricAccumulator:
    """Collect logits/targets and compute one-vs-rest classification metrics."""

    def __init__(
        self,
        num_classes: int,
        undefined_value: float = 0.0,
        log_per_class: bool = True,
        outputs: Mapping[str, Any] | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.undefined_value = float(undefined_value)
        self.default_log_per_class = log_per_class
        self.outputs = dict(outputs or {})
        self.default_enabled = bool(self.outputs.get("default_enabled", True))
        self.default_metrics = self._normalize_metric_names(
            self.outputs.get("default_metrics", ["all"])
        )
        self.output_overrides = dict(self.outputs.get("per_output", {}))
        self._logits: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._targets: dict[str, list[torch.Tensor]] = defaultdict(list)

    def update(self, output_name: str, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Store one batch of logits for one named output."""
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            return
        self._logits[output_name].append(logits.detach().cpu())
        self._targets[output_name].append(target.detach().view(-1).cpu())

    def compute(self) -> Dict[str, float]:
        """Return flattened metric names for every collected output."""
        return self.compute_from_state(self.state())

    def state(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return concatenated logits/targets for every collected output."""
        state = {}
        for output_name in sorted(self._logits):
            state[output_name] = (
                torch.cat(self._logits[output_name], dim=0),
                torch.cat(self._targets[output_name], dim=0).long(),
            )
        return state

    def compute_from_state(
        self,
        state: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, float]:
        """Compute metrics from an explicit output -> logits/targets mapping."""
        results: Dict[str, float] = {}
        for output_name in sorted(state):
            if not self._output_enabled(output_name):
                continue
            logits, targets = state[output_name]
            output_metrics = self._compute_one_output(
                logits,
                targets,
                metric_names=self._metric_names_for_output(output_name),
                log_per_class=self._log_per_class_for_output(output_name),
            )
            for key, value in output_metrics.items():
                results[f"{output_name}/{key}"] = value
        return results

    def reset(self) -> None:
        self._logits.clear()
        self._targets.clear()

    def _output_enabled(self, output_name: str) -> bool:
        override = self.output_overrides.get(output_name, {})
        return bool(override.get("enabled", self.default_enabled))

    def _metric_names_for_output(self, output_name: str) -> set[str]:
        override = self.output_overrides.get(output_name, {})
        return self._normalize_metric_names(
            override.get("metrics", self.default_metrics)
        )

    def _log_per_class_for_output(self, output_name: str) -> bool:
        override = self.output_overrides.get(output_name, {})
        return bool(override.get("log_per_class", self.default_log_per_class))

    def _normalize_metric_names(self, names) -> set[str]:
        if names is None:
            names = ["all"]
        if isinstance(names, str):
            names = [names]
        normalized = {str(name).lower() for name in names}
        if "all" in normalized:
            return {
                "accuracy",
                "auroc",
                "auprc",
                "precision",
                "recall",
                "f1",
                "specificity",
                "tp",
                "tn",
                "fp",
                "fn",
            }
        known = self._normalize_metric_names(["all"])
        unknown = sorted(normalized - known)
        if unknown:
            raise ValueError(f"Unknown metric name(s): {unknown}. Known: {sorted(known)}")
        return normalized

    def _safe(self, value: float) -> float:
        if np.isnan(value) or np.isinf(value):
            return self.undefined_value
        return float(value)

    def _compute_one_output(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        metric_names: set[str],
        log_per_class: bool,
    ) -> Dict[str, float]:
        probs = F.softmax(logits, dim=-1).numpy()
        target_np = targets.numpy()
        pred_np = probs.argmax(axis=1)
        sample_count = max(int(target_np.shape[0]), 1)

        metrics: Dict[str, float] = {}
        if "accuracy" in metric_names:
            metrics["accuracy"] = float((pred_np == target_np).sum() / sample_count)
        per_class = []
        for class_id in range(self.num_classes):
            actual_pos = target_np == class_id
            pred_pos = pred_np == class_id
            tp = int(np.logical_and(actual_pos, pred_pos).sum())
            tn = int(np.logical_and(~actual_pos, ~pred_pos).sum())
            fp = int(np.logical_and(~actual_pos, pred_pos).sum())
            fn = int(np.logical_and(actual_pos, ~pred_pos).sum())

            precision = self._divide(tp, tp + fp)
            recall = self._divide(tp, tp + fn)
            specificity = self._divide(tn, tn + fp)
            f1 = self._divide(2 * precision * recall, precision + recall)
            auroc = (
                self._auroc(actual_pos.astype(int), probs[:, class_id])
                if "auroc" in metric_names
                else self.undefined_value
            )
            auprc = (
                self._auprc(actual_pos.astype(int), probs[:, class_id])
                if "auprc" in metric_names
                else self.undefined_value
            )

            class_metrics = {
                "auroc": auroc,
                "auprc": auprc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "tp": float(tp),
                "tn": float(tn),
                "fp": float(fp),
                "fn": float(fn),
            }
            per_class.append(class_metrics)
            if log_per_class:
                for key, value in class_metrics.items():
                    if key in metric_names:
                        metrics[f"class_{class_id}/{key}"] = self._safe(value)

        for key in ("auroc", "auprc", "precision", "recall", "f1", "specificity"):
            if key in metric_names:
                metrics[f"{key}_macro"] = self._safe(
                    float(np.mean([c[key] for c in per_class]))
                )
        for key in ("tp", "tn", "fp", "fn"):
            if key in metric_names:
                metrics[f"{key}_total"] = float(sum(c[key] for c in per_class))
        return metrics

    def _divide(self, numerator: float, denominator: float) -> float:
        if denominator == 0:
            return self.undefined_value
        return float(numerator / denominator)

    def _auroc(self, binary_target: np.ndarray, score: np.ndarray) -> float:
        if len(np.unique(binary_target)) < 2:
            return self.undefined_value
        try:
            return self._safe(float(roc_auc_score(binary_target, score)))
        except ValueError:
            return self.undefined_value

    def _auprc(self, binary_target: np.ndarray, score: np.ndarray) -> float:
        if binary_target.sum() == 0:
            return self.undefined_value
        try:
            return self._safe(float(average_precision_score(binary_target, score)))
        except ValueError:
            return self.undefined_value
