"""Modular, configurable data-loading pipeline for the AmTICIS DSA dataset.

Public API:
    * :class:`~amticis_pipeline.dataset_loader.AmTICISDataModule` -- Lightning
      ``LightningDataModule`` (the entry point).
    * :class:`~amticis_pipeline.dataset.AmTICISDataset` -- underlying dataset.
    * :func:`~amticis_pipeline.config.load_config` -- YAML config loader.

The pipeline reads raw NIfTI scans and reproduces the preprocessing used by the
repository's original ``Data/dataset.py`` dataloader, while exposing every
hyperparameter through ``config.yaml``.
"""

from .config import PipelineConfig, load_config
from .dataset import AmTICISDataset, ViewSpec
from .dataset_loader import AmTICISDataModule
from .transforms import ComposeTransforms, ResizeView, build_transform_pipeline

__all__ = [
    "AmTICISDataModule",
    "AmTICISDataset",
    "ViewSpec",
    "PipelineConfig",
    "load_config",
    "ComposeTransforms",
    "ResizeView",
    "build_transform_pipeline",
]
