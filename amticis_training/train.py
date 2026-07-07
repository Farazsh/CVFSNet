"""Command-line entrypoint for Lightning-based AmTICIS training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
except Exception:  # pragma: no cover - older installs expose pytorch_lightning
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

from amticis_pipeline.dataset_loader import AmTICISDataModule

from .config import build_pipeline_config, load_training_config, parse_overrides
from .lightning_module import AmTICISLightningModule
from .losses import build_loss
from .models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Training config YAML path.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set trainer.max_epochs=2.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved training config and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config, overrides=parse_overrides(args.set))
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return

    seed_everything(int(config.get("seed", 1)), workers=True)
    pipeline_config = build_pipeline_config(config)
    datamodule = AmTICISDataModule(config=pipeline_config)

    model = build_model(config["model"], pipeline_config)
    criterion = build_loss(config["loss"], pipeline_config.data.num_classes)
    lightning_module = AmTICISLightningModule(
        model=model,
        criterion=criterion,
        config=config,
        num_classes=pipeline_config.data.num_classes,
    )

    run_dir = _run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(run_dir, config)

    trainer = Trainer(
        **dict(config["trainer"]),
        default_root_dir=str(run_dir),
        logger=_build_loggers(config, run_dir),
        callbacks=_build_callbacks(config),
        enable_checkpointing=bool(config.get("checkpointing", {}).get("enabled", True)),
    )
    trainer.fit(lightning_module, datamodule=datamodule)


def _run_dir(config: Dict[str, Any]) -> Path:
    run_cfg = config["run"]
    return Path(run_cfg.get("output_dir", "output_runs_lightning")) / run_cfg["name"]


def _write_resolved_config(run_dir: Path, config: Dict[str, Any]) -> None:
    with open(run_dir / "resolved_training_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _build_loggers(config: Dict[str, Any], run_dir: Path):
    logging_cfg = config.get("logging", {})
    loggers = []

    if logging_cfg.get("csv", {}).get("enabled", True):
        loggers.append(CSVLogger(save_dir=str(run_dir), name="csv"))

    wandb_cfg = logging_cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        api_key_env = wandb_cfg.get("api_key_env")
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if api_key:
            import wandb

            wandb.login(key=api_key)
        if wandb_cfg.get("offline", False):
            os.environ["WANDB_MODE"] = "offline"
        loggers.append(
            WandbLogger(
                project=wandb_cfg.get("project"),
                entity=wandb_cfg.get("entity"),
                name=config["run"]["name"],
                save_dir=str(run_dir),
                tags=wandb_cfg.get("tags"),
                log_model=wandb_cfg.get("log_model", False),
                config=config,
            )
        )

    return loggers if loggers else False


def _build_callbacks(config: Dict[str, Any]):
    callbacks = []
    checkpoint_cfg = config.get("checkpointing", {})
    if checkpoint_cfg.get("enabled", True):
        callbacks.append(
            ModelCheckpoint(
                monitor=checkpoint_cfg.get("monitor", "val/fuse/f1_macro"),
                mode=checkpoint_cfg.get("mode", "max"),
                save_top_k=checkpoint_cfg.get("save_top_k", 3),
                save_last=checkpoint_cfg.get("save_last", True),
                filename="epoch={epoch:03d}",
                auto_insert_metric_name=False,
            )
        )

    early_cfg = config.get("early_stopping", {})
    if early_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=early_cfg.get("monitor", "val/fuse/f1_macro"),
                mode=early_cfg.get("mode", "max"),
                patience=early_cfg.get("patience", 70),
            )
        )

    if config.get("scheduler", {}).get("name"):
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    return callbacks


if __name__ == "__main__":
    main()
