"""Entry point for VideoMAE training (ordinal and binary).

    uv run python -m mae_ordinal.train
    uv run python -m mae_ordinal.train --config mae_ordinal/config_binary_ap.yaml

Reuses the existing data pipeline and metric accumulator; does not modify any
file inside amticis_pipeline / amticis_training / Src / Data / Loss / Lib.
"""

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
except Exception:  # pragma: no cover
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

from amticis_training.config import deep_update, parse_overrides

from .binary_lightning_module import VideoMAEBinaryModule
from .dataloader import build_datamodule
from .lightning_module import VideoMAEOrdinalModule
from .model import build_videomae_binary, build_videomae_ordinal

BINARY_MODEL_NAMES = {"videomae_binary", "videomae_binary_dual"}

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from the repo-root .env into the environment."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except Exception:
        pass
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def load_config(path: Path, overrides) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    deep_update(config, parse_overrides(overrides))
    return config


def main() -> None:
    args = parse_args()
    _load_dotenv()
    config = load_config(args.config, args.set)
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return

    seed_everything(int(config.get("seed", 1)), workers=True)

    data_cfg = config["data"]
    datamodule = build_datamodule(
        pipeline_overrides=data_cfg.get("pipeline_overrides"),
        config_path=data_cfg.get("config_path", "amticis_pipeline/config.yaml"),
        project_root=data_cfg.get("project_root", "."),
        preset=data_cfg.get("preset"),
    )
    num_classes = datamodule.config.data.num_classes

    model_name = str(config.get("model", {}).get("name", "videomae_ordinal"))
    if model_name in BINARY_MODEL_NAMES:
        model = build_videomae_binary(config["model"])
        module = VideoMAEBinaryModule(model=model, config=config, num_classes=num_classes)
    else:
        model = build_videomae_ordinal(config["model"], num_classes)
        module = VideoMAEOrdinalModule(model=model, config=config, num_classes=num_classes)

    run_dir = Path(config["run"].get("output_dir", "output_runs_lightning")) / config["run"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "resolved_training_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    trainer = Trainer(
        **dict(config["trainer"]),
        default_root_dir=str(run_dir),
        logger=_build_loggers(config, run_dir),
        callbacks=_build_callbacks(config),
        enable_checkpointing=bool(config.get("checkpointing", {}).get("enabled", True)),
    )
    trainer.fit(module, datamodule=datamodule)


def _build_loggers(config: Dict[str, Any], run_dir: Path):
    logging_cfg = config.get("logging", {})
    loggers = []
    if logging_cfg.get("csv", {}).get("enabled", True):
        loggers.append(CSVLogger(save_dir=str(run_dir), name="csv"))

    wandb_cfg = logging_cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        api_key = os.environ.get(wandb_cfg.get("api_key_env", "WANDB_API_KEY"))
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
                save_top_k=checkpoint_cfg.get("save_top_k", 1),
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
                patience=early_cfg.get("patience", 40),
            )
        )
    if config.get("scheduler", {}).get("name"):
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    return callbacks


if __name__ == "__main__":
    main()
