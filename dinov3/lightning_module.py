"""Lightning training module for fully fine-tuned DINOv3 binary classification."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

try:
    from lightning.pytorch import LightningModule
except Exception:  # pragma: no cover
    from pytorch_lightning import LightningModule

from .metrics import binary_metrics


class DINOv3LightningModule(LightningModule):
    def __init__(self, model: nn.Module, loss_fn: nn.Module, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.config = dict(config)
        self.save_hyperparameters({"config": self.config}, ignore=["model", "loss_fn"])
        self._train_probabilities: list[torch.Tensor] = []
        self._train_targets: list[torch.Tensor] = []
        self._val_probabilities: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)

    def _step(self, batch: Mapping[str, Any], stage: str) -> torch.Tensor:
        pixels = batch["pixel_values"].to(self.device, non_blocking=True)
        targets = batch["label"].to(self.device, non_blocking=True).view(-1)
        logits = self(pixels).view(-1)
        loss = self.loss_fn(logits, targets.float())
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=int(targets.shape[0]),
        )
        probabilities = torch.sigmoid(logits).detach().cpu()
        if stage == "train":
            self._train_probabilities.append(probabilities)
            self._train_targets.append(targets.detach().cpu())
        else:
            self._val_probabilities.append(probabilities)
            self._val_targets.append(targets.detach().cpu())
        return loss

    def training_step(self, batch, batch_idx):
        del batch_idx
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        del batch_idx
        self._step(batch, "val")

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
        groups = self.model.parameter_groups(
            backbone_lr=float(optimizer_cfg["backbone_lr"]),
            head_lr=float(optimizer_cfg["head_lr"]),
            weight_decay=float(optimizer_cfg["weight_decay"]),
        )
        optimizer = torch.optim.AdamW(
            groups,
            betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
        )
        scheduler_cfg = self.config["scheduler"]
        warmup = max(1, int(scheduler_cfg.get("warmup_epochs", 5)))
        max_epochs = int(self.config["trainer"]["max_epochs"])
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(scheduler_cfg.get("warmup_start_factor", 0.01)),
            total_iters=warmup,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, max_epochs - warmup),
            eta_min=float(scheduler_cfg.get("eta_min", 1e-7)),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
