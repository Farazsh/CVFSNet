"""CORN ordinal-regression utilities (Shi, Cao & Raschka, 2021).

A K-class ordinal target is modelled with ``K - 1`` binary "rank" tasks. The
network emits ``K - 1`` logits per sample; task ``k`` predicts the *conditional*
probability ``P(y > k | y > k-1)``. This yields rank-consistent predictions
without CORAL's shared-weight constraint.

This module provides:
    * :func:`corn_loss`                 -- the conditional training loss;
    * :func:`corn_cumulative_probs`     -- ``P(y > k)`` for k = 0 .. K-2;
    * :func:`corn_class_probs`          -- full categorical ``P(y = k)``;
    * :func:`corn_predict`              -- rank-consistent hard label.

Nothing here depends on the rest of the repo, so it is trivially unit-testable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def corn_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Conditional ordinal loss over ``num_classes - 1`` binary tasks.

    Args:
        logits: ``(N, num_classes - 1)`` raw scores (one per rank threshold).
        targets: ``(N,)`` integer class labels in ``[0, num_classes - 1]``.
        num_classes: total number of ordinal classes ``K``.

    Only samples with label ``> k - 1`` contribute to task ``k`` (the CORN
    conditional-subset trick), so the loss is averaged over the total number of
    contributing (sample, task) pairs.
    """
    if logits.ndim != 2 or logits.shape[1] != num_classes - 1:
        raise ValueError(
            f"Expected logits of shape (N, {num_classes - 1}), got {tuple(logits.shape)}."
        )
    targets = targets.view(-1).long()
    device = logits.device

    total = logits.new_zeros(())
    num_terms = 0
    for task_index in range(num_classes - 1):
        # Conditioning subset: samples still "alive" at this threshold (y > task_index - 1).
        subset_mask = targets > (task_index - 1)
        if subset_mask.sum() == 0:
            continue
        task_logits = logits[subset_mask, task_index]
        task_labels = (targets[subset_mask] > task_index).float().to(device)
        # Binary cross entropy with logits, summed then normalised by total terms.
        total = total + F.binary_cross_entropy_with_logits(
            task_logits, task_labels, reduction="sum"
        )
        num_terms += int(task_logits.shape[0])

    if num_terms == 0:
        return total
    return total / num_terms


def corn_cumulative_probs(logits: torch.Tensor) -> torch.Tensor:
    """Return ``P(y > k)`` for k = 0 .. K-2 as ``cumprod`` of conditionals."""
    conditionals = torch.sigmoid(logits)
    return torch.cumprod(conditionals, dim=1)


def corn_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORN logits ``(N, K-1)`` to categorical probs ``(N, K)``.

    ``P(y=0) = 1 - P(y>0)``; ``P(y=k) = P(y>k-1) - P(y>k)``;
    ``P(y=K-1) = P(y>K-2)``. Rows sum to 1 by construction.
    """
    cum = corn_cumulative_probs(logits)  # (N, K-1) = P(y > k)
    n, km1 = cum.shape
    probs = logits.new_zeros((n, km1 + 1))
    probs[:, 0] = 1.0 - cum[:, 0]
    if km1 > 1:
        probs[:, 1:km1] = cum[:, : km1 - 1] - cum[:, 1:]
    probs[:, km1] = cum[:, -1]
    return probs.clamp(min=0.0)


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """Rank-consistent hard prediction: count thresholds with ``P(y>k) > 0.5``."""
    cum = corn_cumulative_probs(logits)
    return (cum > 0.5).sum(dim=1).long()


def class_pseudologits_from_corn(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Log-probabilities ``(N, K)`` usable as "logits" for the repo metrics.

    The existing ``ClassificationMetricAccumulator`` applies ``softmax`` to the
    tensor it is given; ``softmax(log p) == p``, so feeding log-probs makes it
    compute metrics on the exact CORN categorical probabilities (argmax,
    per-class AUROC/AUPRC, etc.).
    """
    probs = corn_class_probs(logits).clamp(min=eps)
    return torch.log(probs)
