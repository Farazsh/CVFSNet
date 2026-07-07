"""Model registry used by the Lightning training package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import torch.nn as nn

from Src import CVFSNet
from amticis_pipeline.config import PipelineConfig


ModelBuilder = Callable[[Mapping[str, Any], PipelineConfig], nn.Module]
MODEL_REGISTRY: Dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Register a model builder under a config-facing name."""
    normalized = name.lower()

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if normalized in MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[normalized] = builder
        return builder

    return decorator


def _resolve_optional_path(value: Any) -> Any:
    if value in (None, "", False):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str((Path.cwd() / path).resolve())


@register_model("cvfsnet")
def _build_cvfsnet(config: Mapping[str, Any], data_config: PipelineConfig) -> nn.Module:
    """Build CVFSNet using data-derived defaults and config kwargs."""
    params = dict(config.get("params", {}))
    data_num_classes = int(data_config.data.num_classes)
    configured_num_classes = params.get("model_num_class")
    if configured_num_classes is not None and int(configured_num_classes) != data_num_classes:
        raise ValueError(
            "model.params.model_num_class does not match the data label mode: "
            f"got model_num_class={configured_num_classes}, "
            f"but data.label_mode implies num_classes={data_num_classes}. "
            "Set model.params.model_num_class to match data.label_mode, "
            "or remove model_num_class to use the data-derived default."
        )
    params.setdefault("model_num_class", data_num_classes)
    params.setdefault("input_clip_length", data_config.data.num_frames)
    params.setdefault("input_crop_size", data_config.data.image_size)
    params.setdefault("input_channel", 1)

    for key in (
        "all_pretrained",
        "cor_pretrained",
        "fuse_pretrained",
        "sag_pretrained",
    ):
        params[key] = _resolve_optional_path(params.get(key))

    return CVFSNet(**params)


def build_model(config: Mapping[str, Any], data_config: PipelineConfig) -> nn.Module:
    """Build a registered model from the training config."""
    name = str(config["name"]).lower()
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError as error:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Registered models: {known}.") from error
    return builder(config, data_config)


def parameter_groups(
    model: nn.Module,
    base_lr: float,
    cvafm_lr_multiplier: float | None = None,
):
    """Return optimizer parameter groups, optionally scaling CVFM attention LR."""
    if not cvafm_lr_multiplier or cvafm_lr_multiplier == 1:
        return model.parameters()
    try:
        cvafm_params = list(model.fusion_model.CVAFM.parameters())
    except AttributeError:
        return model.parameters()

    cvafm_ids = {id(param) for param in cvafm_params}
    base_params = [p for p in model.parameters() if id(p) not in cvafm_ids]
    scaled_params = [p for p in model.parameters() if id(p) in cvafm_ids]
    return [
        {"params": base_params, "lr": base_lr},
        {"params": scaled_params, "lr": base_lr * cvafm_lr_multiplier},
    ]
