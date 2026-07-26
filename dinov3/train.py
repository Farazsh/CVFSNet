"""Train DINOv3 and evaluate a freshly reloaded best checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv

try:
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
except Exception:  # pragma: no cover
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

from amticis_training.config import parse_overrides

from .dataloader import DINOv3DataModule
from .evaluate import build_model, evaluate_checkpoint, load_config, load_processor
from .lightning_module import DINOv3LightningModule
from .losses import build_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def run_dir(config: Mapping[str, Any]) -> Path:
    return Path(config["run"]["output_dir"]) / str(config["run"]["name"])


def build_loggers(config: Mapping[str, Any], output_dir: Path):
    logging_cfg = config.get("logging", {})
    loggers = []
    if logging_cfg.get("csv", {}).get("enabled", True):
        loggers.append(CSVLogger(save_dir=str(output_dir), name="csv"))
    wandb_cfg = logging_cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        api_key = os.environ.get(str(wandb_cfg.get("api_key_env", "WANDB_API_KEY")))
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
                save_dir=str(output_dir),
                tags=wandb_cfg.get("tags"),
                log_model=wandb_cfg.get("log_model", False),
                config=dict(config),
            )
        )
    return loggers if loggers else False


def build_callbacks(config: Mapping[str, Any], output_dir: Path):
    checkpoint_cfg = config["checkpointing"]
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="epoch={epoch:03d}-val_auroc={val/auroc:.4f}",
        auto_insert_metric_name=False,
        monitor=checkpoint_cfg["monitor"],
        mode=checkpoint_cfg.get("mode", "max"),
        save_top_k=int(checkpoint_cfg.get("save_top_k", 1)),
        save_last=bool(checkpoint_cfg.get("save_last", True)),
    )
    callbacks: list[Any] = [checkpoint, LearningRateMonitor(logging_interval="epoch")]
    early_cfg = config.get("early_stopping", {})
    if early_cfg.get("enabled", True):
        callbacks.append(
            EarlyStopping(
                monitor=early_cfg.get("monitor", checkpoint_cfg["monitor"]),
                mode=early_cfg.get("mode", "max"),
                patience=int(early_cfg.get("patience", 15)),
            )
        )
    return callbacks, checkpoint


def main() -> None:
    args = parse_args()
    config = load_config(args.config, parse_overrides(args.set))
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    token = os.environ.get(str(config["model"].get("token_env", "HF_TOKEN")))
    _, mean, std = load_processor(config, token)

    seed_everything(int(config["seed"]), workers=True)
    output_dir = run_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)

    datamodule = DINOv3DataModule(config["data"], mean, std, project_root=project_root)
    model = build_model(config, token)
    module = DINOv3LightningModule(model, build_loss(config["loss"]), config)
    callbacks, checkpoint_callback = build_callbacks(config, output_dir)
    trainer = Trainer(
        **dict(config["trainer"]),
        default_root_dir=str(output_dir),
        logger=build_loggers(config, output_dir),
        callbacks=callbacks,
        enable_checkpointing=True,
    )
    trainer.fit(module, datamodule=datamodule)

    best_path = checkpoint_callback.best_model_path
    if not best_path:
        raise RuntimeError("Training completed without a best checkpoint path.")
    result = evaluate_checkpoint(config, best_path, output_dir)
    print(f"Best checkpoint: {best_path}")
    print(f"Tuned threshold: {result['threshold']:.6f}")
    print(f"Workbook: {result['paths']['workbook']}")


if __name__ == "__main__":
    main()
