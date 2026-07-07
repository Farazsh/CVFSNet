"""Loss registry for Lightning training.

The default loss is a Lightning-safe implementation of the legacy
``LabelSmoothSeasaw_MISO`` objective. It keeps the same multi-output weighting
concept while avoiding hard-coded ``.cuda()`` state during construction.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


LossBuilder = Callable[[Mapping[str, Any], int], nn.Module]
LOSS_REGISTRY: Dict[str, LossBuilder] = {}


def register_loss(name: str) -> Callable[[LossBuilder], LossBuilder]:
    normalized = name.lower()

    def decorator(builder: LossBuilder) -> LossBuilder:
        if normalized in LOSS_REGISTRY:
            raise KeyError(f"Loss '{name}' is already registered.")
        LOSS_REGISTRY[normalized] = builder
        return builder

    return decorator


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with uniform label smoothing."""

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1).")
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.view(-1).long()
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return (self.confidence * nll + self.smoothing * smooth).mean()


class SeesawCrossEntropy(nn.Module):
    """Seesaw cross entropy for long-tailed single-label classification."""

    def __init__(
        self,
        num_classes: int,
        p: float = 0.8,
        q: float = 2.0,
        eps: float = 1e-2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.eps = eps
        self.register_buffer("cum_samples", torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.view(-1).long()
        if logits.size(-1) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} logits, got {logits.size(-1)}."
            )

        with torch.no_grad():
            unique_labels = target.unique()
            for label in unique_labels:
                self.cum_samples[label] += (target == label).sum()

        one_hot = F.one_hot(target, self.num_classes)
        weights = logits.new_ones(one_hot.size())

        if self.p > 0:
            sample_ratio = self.cum_samples[None, :].clamp(min=1) / self.cum_samples[
                :, None
            ].clamp(min=1)
            index = (sample_ratio < 1.0).float()
            mitigation = sample_ratio.pow(self.p) * index + (1 - index)
            weights = weights * mitigation[target, :]

        if self.q > 0:
            scores = F.softmax(logits.detach(), dim=1)
            self_scores = scores[torch.arange(len(scores), device=scores.device), target]
            score_ratio = scores / self_scores[:, None].clamp(min=self.eps)
            index = (score_ratio > 1.0).float()
            compensation = score_ratio.pow(self.q) * index + (1 - index)
            weights = weights * compensation

        adjusted_logits = logits + weights.log() * (1 - one_hot)
        return F.cross_entropy(adjusted_logits, target)


class LabelSmoothSeesawMISOLoss(nn.Module):
    """Multi-output label-smoothing + seesaw loss.

    The model is expected to return a list/tuple where each tensor has shape
    ``(batch, num_classes)``. ``None`` outputs are skipped, matching CVFSNet's
    optional deep supervision heads.
    """

    def __init__(
        self,
        num_classes: int,
        balance: Iterable[Iterable[float]],
        smoothing: float = 0.4,
        p: float = 0.8,
        q: float = 1.0,
        eps: float = 1e-2,
    ) -> None:
        super().__init__()
        self.balance = [list(row) for row in balance]
        self.label_smooth = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.seesaw = SeesawCrossEntropy(num_classes=num_classes, p=p, q=q, eps=eps)

    def forward(self, outputs, target: torch.Tensor):
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]

        total = None
        loss_dict: Dict[str, torch.Tensor] = {}
        for index, logits in enumerate(outputs):
            if logits is None or not torch.is_tensor(logits) or logits.ndim != 2:
                continue
            weights = self.balance[index] if index < len(self.balance) else [1.0, 1.0]
            smooth_loss = self.label_smooth(logits, target) * float(weights[0])
            seesaw_loss = self.seesaw(logits, target) * float(weights[1])
            subtotal = smooth_loss + seesaw_loss
            total = subtotal if total is None else total + subtotal
            phase = _output_phase_name(index)
            loss_dict[f"LS#{phase}"] = smooth_loss.detach()
            loss_dict[f"SS#{phase}"] = seesaw_loss.detach()

        if total is None:
            raise ValueError("No tensor logits were provided to the loss.")
        loss_dict["loss"] = total.detach()
        return total, loss_dict


def _output_phase_name(index: int) -> str:
    if index == 0:
        return "FUSE"
    if index == 1:
        return "AP"
    if index == 2:
        return "SAG"
    return str(index)


@register_loss("label_smooth_seesaw_miso")
def _build_label_smooth_seesaw(
    config: Mapping[str, Any],
    num_classes: int,
) -> nn.Module:
    params = dict(config.get("params", {}))
    params.setdefault("num_classes", num_classes)
    return LabelSmoothSeesawMISOLoss(**params)


@register_loss("cross_entropy_miso")
def _build_cross_entropy_miso(
    config: Mapping[str, Any],
    num_classes: int,
) -> nn.Module:
    del num_classes
    weights = list(config.get("params", {}).get("output_weights", [1.0]))
    return MultiOutputCrossEntropy(weights)


class MultiOutputCrossEntropy(nn.Module):
    """Simple weighted cross entropy across every tensor output."""

    def __init__(self, output_weights: list[float]) -> None:
        super().__init__()
        self.output_weights = output_weights

    def forward(self, outputs, target: torch.Tensor):
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        total = None
        loss_dict: Dict[str, torch.Tensor] = {}
        for index, logits in enumerate(outputs):
            if logits is None or not torch.is_tensor(logits) or logits.ndim != 2:
                continue
            weight = self.output_weights[index] if index < len(self.output_weights) else 1
            loss = F.cross_entropy(logits, target.view(-1).long()) * float(weight)
            total = loss if total is None else total + loss
            loss_dict[f"CE#{_output_phase_name(index)}"] = loss.detach()
        if total is None:
            raise ValueError("No tensor logits were provided to the loss.")
        loss_dict["loss"] = total.detach()
        return total, loss_dict


def build_loss(config: Mapping[str, Any], num_classes: int) -> nn.Module:
    name = str(config["name"]).lower()
    try:
        builder = LOSS_REGISTRY[name]
    except KeyError as error:
        known = ", ".join(sorted(LOSS_REGISTRY))
        raise KeyError(f"Unknown loss '{name}'. Registered losses: {known}.") from error
    return builder(config, num_classes)
