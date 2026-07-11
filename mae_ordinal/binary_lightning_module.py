"""Lightning module for VideoMAE binary T012a vs T2b3 classification."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from lightning.pytorch import LightningModule
except Exception:  # pragma: no cover
    from pytorch_lightning import LightningModule

from amticis_training.classification_metrics import ClassificationMetricAccumulator

from .model import binary_pseudologits, layerwise_param_groups

OUTPUT_NAME = "fuse"


class VideoMAEBinaryModule(LightningModule):
    def __init__(self, model: nn.Module, config: Mapping[str, Any], num_classes: int) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError(f"Binary module expects num_classes=2, got {num_classes}.")
        self.model = model
        self.config = dict(config)
        self.num_classes = num_classes
        input_cfg = self.config.get("model", {}).get("input", {})
        self.dual_view = input_cfg.get("type") == "dual_view_list"
        self.view_key = input_cfg.get("view", "AP")

        metric_cfg = self.config.get("metrics", {})
        acc_kwargs = dict(
            num_classes=num_classes,
            undefined_value=metric_cfg.get("undefined_value", 0.0),
            log_per_class=metric_cfg.get("log_per_class", True),
            outputs=metric_cfg.get("outputs"),
        )
        self.train_metrics = ClassificationMetricAccumulator(**acc_kwargs)
        self.val_metrics = ClassificationMetricAccumulator(**acc_kwargs)

    def _model_input(self, batch: Mapping[str, Any]):
        if self.dual_view:
            return [
                batch["AP"].to(self.device, non_blocking=True).contiguous(),
                batch["sagittal"].to(self.device, non_blocking=True).contiguous(),
            ]
        return batch[self.view_key].to(self.device, non_blocking=True).contiguous()

    def _shared_step(self, batch: Mapping[str, Any], stage: str) -> torch.Tensor:
        model_input = self._model_input(batch)
        label = batch["label"].to(self.device, non_blocking=True).view(-1).long()
        logit = self.model(model_input).view(-1)
        loss = F.binary_cross_entropy_with_logits(logit, label.float())

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=int(label.shape[0]),
        )
        accumulator = self.train_metrics if stage == "train" else self.val_metrics
        pseudo = binary_pseudologits(logit.detach().unsqueeze(1))
        accumulator.update(OUTPUT_NAME, pseudo, label)
        return loss

    def on_train_epoch_start(self) -> None:
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
        metrics = accumulator.compute()
        loggable = {
            f"{stage}/{key}": torch.tensor(float(value), device=self.device)
            for key, value in metrics.items()
        }
        if loggable:
            self.log_dict(loggable, on_epoch=True, prog_bar=False)
        accumulator.reset()

    def configure_optimizers(self):
        optimizer_cfg = self.config["optimizer"]
        base_lr = float(
            optimizer_cfg.get("base_lr", optimizer_cfg.get("params", {}).get("lr", 1e-4))
        )
        weight_decay = float(optimizer_cfg.get("weight_decay", 0.05))
        layer_decay = optimizer_cfg.get("layer_decay")

        if layer_decay:
            groups = layerwise_param_groups(
                self.model, base_lr=base_lr, weight_decay=weight_decay, layer_decay=float(layer_decay)
            )
        else:
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
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            max_epochs = int(self.config["trainer"]["max_epochs"])
            warmup = max(1, int(scheduler_cfg.get("warmup_epochs", 5)))
            eta_min = float(scheduler_cfg.get("eta_min", 1e-6))
            warmup_sched = LinearLR(
                optimizer,
                start_factor=float(scheduler_cfg.get("warmup_start_factor", 0.01)),
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
