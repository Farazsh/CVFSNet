"""Loss construction for DINOv3 binary mTICI experiments."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


def build_loss(config: Mapping[str, Any]) -> nn.Module:
    name = str(config.get("name", "bce_with_logits")).lower()
    if name != "bce_with_logits":
        raise ValueError(f"Unsupported DINOv3 loss '{name}'.")
    value = config.get("pos_weight")
    pos_weight = None if value is None else torch.tensor([float(value)], dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
