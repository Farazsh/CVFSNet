"""Coronal-only (AP) data module built on the existing ``amticis_pipeline``.

We do NOT reimplement any preprocessing. This module only assembles a
:class:`~amticis_pipeline.config.PipelineConfig` with the coronal single-view
overrides required for the VideoMAE experiment (AP only, 16 frames, 224x224,
fuse01 labels) and hands it to the repo's own
:class:`~amticis_pipeline.dataset_loader.AmTICISDataModule`.

The trilinear resample, the DSA augmentations, the ``TioZNormalization(div255)``
normalization and the ``WeightedRandomSampler`` all come from the existing
pipeline unchanged -- only the frame count / spatial size / active view differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from amticis_pipeline.dataset_loader import AmTICISDataModule
from amticis_training.config import build_pipeline_config

REPO_ROOT = Path(__file__).resolve().parents[1]

# Coronal single-view overrides deep-merged into amticis_pipeline/config.yaml.
# NB: with active=[AP] and AP mapping _C->_C, the loader yields exactly one
# coronal clip per study under batch["AP"], shape (B, 1, T, H, W).
CORONAL_OVERRIDES: Dict[str, Any] = {
    "data": {
        "label_mode": "fuse01",
        "num_frames": 16,
        "image_size": 224,
    },
    "views": {
        "active": ["AP"],
        "definitions": {
            "AP": {"replace_from": "_C", "replace_to": "_C"},
        },
    },
}


def build_datamodule(
    pipeline_overrides: Optional[Dict[str, Any]] = None,
    config_path: str = "amticis_pipeline/config.yaml",
    project_root: str = ".",
) -> AmTICISDataModule:
    """Return an ``AmTICISDataModule`` configured for coronal-only VideoMAE input."""
    overrides = _deep_merge(dict(CORONAL_OVERRIDES), pipeline_overrides or {})
    training_like = {
        "data": {
            "config_path": config_path,
            "project_root": project_root,
            "pipeline_overrides": overrides,
        }
    }
    pipeline_config = build_pipeline_config(training_like)
    return AmTICISDataModule(config=pipeline_config)


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base
