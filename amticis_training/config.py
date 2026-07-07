"""Configuration loading for the Lightning training package.

The training config is intentionally a plain YAML file with a small loader
instead of a large framework-specific schema. The loader validates the few
sections that are required by the trainer and leaves architecture/loss kwargs
as dictionaries so researchers can add new modules without changing config
classes first.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import yaml

from amticis_pipeline.config import (
    DataConfig,
    LoaderConfig,
    PipelineConfig,
    SplitsConfig,
    TransformStep,
    TransformsConfig,
    ViewDefinition,
    ViewsConfig,
)


REQUIRED_TOP_LEVEL = {
    "data",
    "loss",
    "metrics",
    "model",
    "optimizer",
    "run",
    "trainer",
}


def deep_update(
    base: MutableMapping[str, Any],
    updates: Mapping[str, Any] | None,
) -> MutableMapping[str, Any]:
    """Recursively merge ``updates`` into ``base`` and return ``base``."""
    if not updates:
        return base
    for key, value in updates.items():
        if (
            isinstance(value, Mapping)
            and isinstance(base.get(key), MutableMapping)
        ):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _set_by_dotted_key(config: MutableMapping[str, Any], key: str, value: Any) -> None:
    current = config
    parts = key.split(".")
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, MutableMapping):
            raise ValueError(f"Cannot set '{key}': '{part}' is not a mapping.")
        current = next_value
    current[parts[-1]] = value


def parse_overrides(raw_overrides: Iterable[str] | None) -> Dict[str, Any]:
    """Parse CLI overrides of the form ``section.key=value``.

    Values are parsed with ``yaml.safe_load`` so numbers, booleans, nulls, and
    lists can be passed naturally:

    ``--set trainer.max_epochs=2 model.params.deep_super=[false,false,false,false]``
    """
    parsed: Dict[str, Any] = {}
    for override in raw_overrides or []:
        if "=" not in override:
            raise ValueError(
                f"Override '{override}' must use the form dotted.key=value."
            )
        key, value = override.split("=", 1)
        _set_by_dotted_key(parsed, key, yaml.safe_load(value))
    return parsed


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(payload).__name__}.")
    return payload


def load_training_config(
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Load and validate the training YAML config.

    Args:
        config_path: Path to the training config. Defaults to
            ``amticis_training/config.yaml``.
        overrides: Already-parsed overrides to deep-merge after reading YAML.
    """
    path = Path(config_path) if config_path else Path(__file__).with_name("config.yaml")
    config = _read_yaml(path)
    deep_update(config, overrides)
    missing = sorted(REQUIRED_TOP_LEVEL - set(config))
    if missing:
        raise ValueError(f"Training config is missing required sections: {missing}.")
    config["_config_path"] = str(path.resolve())
    return config


def _parse_transform_steps(raw: list[dict[str, Any]]) -> list[TransformStep]:
    return [
        TransformStep(name=step["name"], params=step.get("params", {}) or {})
        for step in raw
    ]


def build_pipeline_config(training_config: Mapping[str, Any]) -> PipelineConfig:
    """Build the AmTICIS data-pipeline config with training-level overrides."""
    data_cfg = training_config["data"]
    config_path = Path(data_cfg.get("config_path", "amticis_pipeline/config.yaml"))
    project_root = Path(data_cfg.get("project_root", ".")).resolve()

    raw = _read_yaml(config_path)
    deep_update(raw, data_cfg.get("pipeline_overrides"))

    views_raw = raw["views"]
    definitions = {
        name: ViewDefinition(**rule)
        for name, rule in views_raw["definitions"].items()
    }
    transforms_raw = raw["transforms"]
    return PipelineConfig(
        data=DataConfig(**raw["data"]),
        views=ViewsConfig(active=list(views_raw["active"]), definitions=definitions),
        splits=SplitsConfig(**raw["splits"]),
        loader=LoaderConfig(**raw["loader"]),
        transforms=TransformsConfig(
            train=_parse_transform_steps(transforms_raw["train"]),
            val=_parse_transform_steps(transforms_raw["val"]),
        ),
        project_root=project_root,
    )
