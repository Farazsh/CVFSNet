"""Lightning module for VideoMAE + CORN ordinal regression.

Reuses the repo's ``ClassificationMetricAccumulator`` verbatim so the logged
metric keys (``val/fuse/f1_macro`` etc.) are identical to the CVFSNet baseline,
and adds ordinal-specific extras (QWK, MAE, off-by-one accuracy).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score

try:
    from lightning.pytorch import LightningModule
except Exception:  # pragma: no cover
    from pytorch_lightning import LightningModule

from amticis_training.classification_metrics import ClassificationMetricAccumulator

from .model import layerwise_param_groups
from .ordinal import class_pseudologits_from_corn, corn_loss

# Single output head, named "fuse" so the primary key is `val/fuse/f1_macro`,
# matching the CVFSNet coronal baseline's checkpoint monitor.
OUTPUT_NAME = "fuse"


class VideoMAEOrdinalModule(LightningModule):
    def __init__(self, model: nn.Module, config: Mapping[str, Any], num_classes: int) -> None:
        super().__init__()
        self.model = model
        self.config = dict(config)
        self.num_classes = num_classes
        self.view_key = self.config.get("model", {}).get("input", {}).get("view", "AP")

        metric_cfg = self.config.get("metrics", {})
        acc_kwargs = dict(
            num_classes=num_classes,
            undefined_value=metric_cfg.get("undefined_value", 0.0),
            log_per_class=metric_cfg.get("log_per_class", True),
            outputs=metric_cfg.get("outputs"),
        )
        self.train_metrics = ClassificationMetricAccumulator(**acc_kwargs)
        self.val_metrics = ClassificationMetricAccumulator(**acc_kwargs)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.model(clip)

    def _shared_step(self, batch: Mapping[str, Any], stage: str) -> torch.Tensor:
        clip = batch[self.view_key].to(self.device, non_blocking=True).contiguous()
        label = batch["label"].to(self.device, non_blocking=True).view(-1).long()
        logits = self(clip)
        loss = corn_loss(logits, label, self.num_classes)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=int(label.shape[0]),
        )
        accumulator = self.train_metrics if stage == "train" else self.val_metrics
        pseudo = class_pseudologits_from_corn(logits).detach()
        accumulator.update(OUTPUT_NAME, pseudo, label)
        return loss

    def on_train_epoch_start(self) -> None:
        # HF `from_pretrained` returns the backbone in eval mode and this
        # Lightning version does not force train mode; set it explicitly so the
        # full fine-tune trains the whole model (dropout/etc. active). The
        # model's overridden .train() keeps a frozen backbone in eval mode.
        self.model.train()

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def on_train_epoch_end(self) -> None:
        self._log_epoch("train", self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch("val", self.val_metrics)

    def _log_epoch(self, stage: str, accumulator: ClassificationMetricAccumulator) -> None:
        state = accumulator.state()
        metrics = accumulator.compute_from_state(state)
        loggable = {
            f"{stage}/{key}": torch.tensor(float(value), device=self.device)
            for key, value in metrics.items()
        }
        loggable.update(self._ordinal_extras(stage, state))
        if loggable:
            self.log_dict(loggable, on_epoch=True, prog_bar=False)
        accumulator.reset()

    def _ordinal_extras(self, stage: str, state) -> dict:
        if OUTPUT_NAME not in state:
            return {}
        pseudo, targets = state[OUTPUT_NAME]
        preds = pseudo.argmax(dim=1).cpu().numpy()
        target_np = targets.cpu().numpy()
        if target_np.size == 0:
            return {}
        mae = float(np.mean(np.abs(preds - target_np)))
        off_by_one = float(np.mean(np.abs(preds - target_np) <= 1))
        try:
            qwk = float(
                cohen_kappa_score(
                    target_np,
                    preds,
                    labels=list(range(self.num_classes)),
                    weights="quadratic",
                )
            )
            if np.isnan(qwk):
                qwk = 0.0
        except ValueError:
            qwk = 0.0
        return {
            f"{stage}/{OUTPUT_NAME}/mae": torch.tensor(mae, device=self.device),
            f"{stage}/{OUTPUT_NAME}/acc_off_by_one": torch.tensor(off_by_one, device=self.device),
            f"{stage}/{OUTPUT_NAME}/qwk": torch.tensor(qwk, device=self.device),
        }

    def configure_optimizers(self):
        optimizer_cfg = self.config["optimizer"]
        base_lr = float(
            optimizer_cfg.get("base_lr", optimizer_cfg.get("params", {}).get("lr", 1e-4))
        )
        weight_decay = float(optimizer_cfg.get("weight_decay", 0.05))
        layer_decay = optimizer_cfg.get("layer_decay")

        if layer_decay:
            # Full fine-tuning: VideoMAE-style layer-wise LR decay.
            groups = layerwise_param_groups(
                self.model, base_lr=base_lr, weight_decay=weight_decay, layer_decay=float(layer_decay)
            )
        else:
            # Frozen / simple: one decayed + one no-decay group over trainable params.
            decay, no_decay = [], []
            for _, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                (no_decay if param.ndim == 1 else decay).append(param)
            groups = [
                {"params": decay, "lr": base_lr, "weight_decay": weight_decay},
                {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
            ]

        betas = tuple(optimizer_cfg.get("betas", (0.9, 0.999)))
        optimizer = torch.optim.AdamW(groups, lr=base_lr, betas=betas)

        scheduler_cfg = self.config.get("scheduler") or {}
        scheduler = self._build_scheduler(optimizer, scheduler_cfg)
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": scheduler_cfg.get("interval", "epoch"),
                "frequency": scheduler_cfg.get("frequency", 1),
                "monitor": scheduler_cfg.get("monitor", "val/fuse/f1_macro"),
            },
        }

    def _build_scheduler(self, optimizer, scheduler_cfg):
        name = scheduler_cfg.get("name")
        if not name:
            return None
        if name == "warmup_cosine":
            from torch.optim.lr_scheduler import (
                CosineAnnealingLR,
                LinearLR,
                SequentialLR,
            )

            max_epochs = int(self.config["trainer"]["max_epochs"])
            warmup = max(1, int(scheduler_cfg.get("warmup_epochs", 5)))
            eta_min = float(scheduler_cfg.get("eta_min", 1e-6))
            warmup_sched = LinearLR(
                optimizer, start_factor=float(scheduler_cfg.get("warmup_start_factor", 0.01)),
                total_iters=warmup,
            )
            cosine_sched = CosineAnnealingLR(
                optimizer, T_max=max(1, max_epochs - warmup), eta_min=eta_min
            )
            return SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup]
            )
        scheduler_cls = getattr(torch.optim.lr_scheduler, name)
        return scheduler_cls(optimizer, **dict(scheduler_cfg.get("params", {})))
