"""Lightning module for VideoMAE binary T012a vs T2b3 classification.

Metrics match ``dinov3``: epoch-level AUROC / AUPRC / accuracy / precision /
recall / F1 / specificity logged as flat ``train|val/<name>`` keys.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from lightning.pytorch import LightningModule
except Exception:  # pragma: no cover
    from pytorch_lightning import LightningModule

from dinov3.metrics import binary_metrics

from .model import layerwise_param_groups


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
        self.save_hyperparameters({"config": self.config}, ignore=["model"])
        self._train_probabilities: list[torch.Tensor] = []
        self._train_targets: list[torch.Tensor] = []
        self._val_probabilities: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

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
        probabilities = torch.sigmoid(logit).detach().cpu()
        if stage == "train":
            self._train_probabilities.append(probabilities)
            self._train_targets.append(label.detach().cpu())
        else:
            self._val_probabilities.append(probabilities)
            self._val_targets.append(label.detach().cpu())
        return loss

    def on_train_epoch_start(self) -> None:
        self.model.train()

    def training_step(self, batch, batch_idx):
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        del batch_idx
        self._shared_step(batch, "val")

    def on_train_epoch_end(self) -> None:
        self._log_metrics("train", self._train_probabilities, self._train_targets)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val", self._val_probabilities, self._val_targets)

    def _log_metrics(
        self,
        stage: str,
        probabilities: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> None:
        if probabilities:
            target_tensor = torch.cat(targets)
            # Lightning sanity checks may inspect only a small, single-class
            # prefix. Full-epoch and final evaluation still require both classes.
            if torch.unique(target_tensor).numel() == 2:
                metrics = binary_metrics(
                    torch.cat(probabilities).numpy(),
                    target_tensor.numpy(),
                    threshold=0.5,
                )
                self.log_dict(
                    {
                        f"{stage}/{name}": torch.tensor(value, device=self.device)
                        for name, value in metrics.items()
                    },
                    on_epoch=True,
                    sync_dist=False,
                )
        probabilities.clear()
        targets.clear()

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
                "monitor": scheduler_cfg.get("monitor", "val/auroc"),
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
