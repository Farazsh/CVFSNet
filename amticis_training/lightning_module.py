"""PyTorch Lightning module for AmTICIS classification."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

try:
    from lightning.pytorch import LightningModule
except Exception:  # pragma: no cover - older installs expose pytorch_lightning
    from pytorch_lightning import LightningModule

from .classification_metrics import ClassificationMetricAccumulator
from .models import parameter_groups


class AmTICISLightningModule(LightningModule):
    """Train/validate a configured classifier on AmTICIS batches."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        config: Mapping[str, Any],
        num_classes: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.config = dict(config)
        self.num_classes = num_classes

        model_cfg = self.config["model"]
        self.input_cfg = dict(model_cfg.get("input", {"type": "dual_view_list"}))
        self.output_names = list(model_cfg.get("output_names", []))

        metric_cfg = self.config.get("metrics", {})
        self.train_metrics = ClassificationMetricAccumulator(
            num_classes=num_classes,
            undefined_value=metric_cfg.get("undefined_value", 0.0),
            log_per_class=metric_cfg.get("log_per_class", True),
            outputs=metric_cfg.get("outputs"),
        )
        self.val_metrics = ClassificationMetricAccumulator(
            num_classes=num_classes,
            undefined_value=metric_cfg.get("undefined_value", 0.0),
            log_per_class=metric_cfg.get("log_per_class", True),
            outputs=metric_cfg.get("outputs"),
        )

    def forward(self, model_input):
        return self.model(model_input)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int):
        del batch_idx
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int):
        del batch_idx
        self._shared_step(batch, stage="val")

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train", self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val", self.val_metrics)

    def configure_optimizers(self):
        optimizer_cfg = self.config["optimizer"]
        optimizer_name = optimizer_cfg["name"]
        optimizer_params = dict(optimizer_cfg.get("params", {}))
        lr = float(optimizer_params.get("lr", 0.001))
        group_cfg = optimizer_cfg.get("parameter_groups", {})
        params = parameter_groups(
            self.model,
            base_lr=lr,
            cvafm_lr_multiplier=group_cfg.get("cvafm_lr_multiplier"),
        )
        optimizer_cls = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_cls(params=params, **optimizer_params)

        scheduler_cfg = self.config.get("scheduler") or {}
        scheduler_name = scheduler_cfg.get("name")
        if not scheduler_name:
            return optimizer

        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
        scheduler = scheduler_cls(optimizer, **dict(scheduler_cfg.get("params", {})))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": scheduler_cfg.get("interval", "epoch"),
                "frequency": scheduler_cfg.get("frequency", 1),
                "monitor": scheduler_cfg.get("monitor", "val/fuse/f1_macro"),
            },
        }

    def _shared_step(self, batch: Mapping[str, Any], stage: str):
        label = batch["label"].to(self.device, non_blocking=True).view(-1).long()
        model_input = self._batch_to_model_input(batch)
        outputs = self(model_input)
        outputs = self._unwrap_outputs(outputs)
        loss, loss_dict = self.criterion(outputs, label)
        batch_size = int(label.shape[0])

        self._log_losses(stage, loss, loss_dict, batch_size=batch_size)
        accumulator = self.train_metrics if stage == "train" else self.val_metrics
        self._record_metrics(accumulator, outputs, label)
        return loss

    def _batch_to_model_input(self, batch: Mapping[str, Any]):
        input_type = self.input_cfg.get("type", "dual_view_list")
        if input_type == "dual_view_list":
            views = self.input_cfg.get("views", ["AP", "sagittal"])
            missing = [view for view in views if view not in batch]
            if missing:
                raise KeyError(
                    f"Batch is missing required view(s) {missing}. "
                    f"Available keys: {list(batch)}."
                )
            return [
                batch[view].to(self.device, non_blocking=True).contiguous()
                for view in views
            ]
        if input_type == "single_view_tensor":
            view = self.input_cfg["view"]
            return batch[view].to(self.device, non_blocking=True).contiguous()
        if input_type == "batch_dict":
            return batch
        raise ValueError(f"Unsupported model input type: {input_type}.")

    def _unwrap_outputs(self, outputs):
        if isinstance(outputs, tuple) and outputs and isinstance(outputs[0], (list, tuple)):
            return list(outputs[0])
        if isinstance(outputs, tuple):
            return list(outputs)
        if isinstance(outputs, list):
            return outputs
        return [outputs]

    def _log_losses(
        self,
        stage: str,
        loss: torch.Tensor,
        loss_dict: Mapping[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        on_step = stage == "train"
        self.log(
            f"{stage}/loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        for name, value in loss_dict.items():
            self.log(
                f"{stage}/loss/{name}",
                value,
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )

    def _record_metrics(
        self,
        accumulator: ClassificationMetricAccumulator,
        outputs,
        label: torch.Tensor,
    ) -> None:
        for index, logits in enumerate(outputs):
            if logits is None or not torch.is_tensor(logits):
                continue
            name = self.output_names[index] if index < len(self.output_names) else f"output_{index}"
            accumulator.update(name, logits, label)

    def _log_epoch_metrics(
        self,
        stage: str,
        accumulator: ClassificationMetricAccumulator,
    ) -> None:
        state = self._gather_metric_state(accumulator.state())
        metrics = accumulator.compute_from_state(state)
        loggable = {
            f"{stage}/{key}": torch.tensor(value, device=self.device)
            for key, value in metrics.items()
        }
        if loggable:
            self.log_dict(loggable, on_epoch=True, prog_bar=False, sync_dist=True)
        accumulator.reset()

    def _gather_metric_state(
        self,
        state: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        world_size = getattr(self.trainer, "world_size", 1) if self.trainer else 1
        if world_size <= 1 or not state:
            return state

        gathered_state = {}
        for output_name, (logits, targets) in state.items():
            gathered_logits = self._all_gather_variable_rows(logits.to(self.device)).cpu()
            gathered_targets = (
                self._all_gather_variable_rows(targets.to(self.device)).cpu().long()
            )
            gathered_state[output_name] = (gathered_logits, gathered_targets)
        return gathered_state

    def _all_gather_variable_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        local_rows = torch.tensor([tensor.shape[0]], device=self.device, dtype=torch.long)
        row_counts = self.all_gather(local_rows).view(-1).long()
        max_rows = int(row_counts.max().item())
        if tensor.shape[0] < max_rows:
            pad_shape = (max_rows - tensor.shape[0],) + tuple(tensor.shape[1:])
            tensor = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)

        gathered = self.all_gather(tensor)
        chunks = [gathered[rank, :rows] for rank, rows in enumerate(row_counts.tolist())]
        return torch.cat(chunks, dim=0)
