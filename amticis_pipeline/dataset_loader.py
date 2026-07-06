"""``AmTICISDataModule`` -- a PyTorch Lightning ``LightningDataModule``.

This is the entry point of the pipeline. It wires the configuration to the
:class:`~amticis_pipeline.dataset.AmTICISDataset` and exposes the standard
Lightning hooks:

    * :meth:`setup`           -- read the split, build per-stage datasets
    * :meth:`train_dataloader`
    * :meth:`val_dataloader`
    * :meth:`test_dataloader`

Run this file directly for a quick smoke test:

    uv run python -m amticis_pipeline.dataset_loader
"""

from __future__ import annotations

from typing import Dict, List, Optional

from torch.utils.data import DataLoader, WeightedRandomSampler

try:  # Lightning >= 2.0 preferred import path, with a fallback for older installs.
    from lightning.pytorch import LightningDataModule
except Exception:  # pragma: no cover
    from pytorch_lightning import LightningDataModule

from .config import PipelineConfig, load_config
from .dataset import AmTICISDataset, ViewSpec
from .transforms import ComposeTransforms, ResizeView, build_transform_pipeline
from .utils import read_split


class AmTICISDataModule(LightningDataModule):
    """Loads raw AmTICIS NIfTI scans with the paper's preprocessing pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        super().__init__()
        self.config = config or load_config()
        self._splits: Dict[str, List[str]] = {}
        self._datasets: Dict[str, AmTICISDataset] = {}

    # -- construction helpers ------------------------------------------------
    def _view_specs(self) -> List[ViewSpec]:
        """Resolve the active view names into concrete file-naming rules."""
        definitions = self.config.views.definitions
        return [
            ViewSpec(
                name=name,
                replace_from=definitions[name].replace_from,
                replace_to=definitions[name].replace_to,
            )
            for name in self.config.views.active
        ]

    def _resize_view(self) -> ResizeView:
        data = self.config.data
        return ResizeView(
            num_frames=data.num_frames,
            image_size=data.image_size,
            interpolation=data.interpolation,
            align_corners=data.align_corners,
        )

    def _build_dataset(
        self, split_key: str, transform: ComposeTransforms
    ) -> AmTICISDataset:
        data = self.config.data
        return AmTICISDataset(
            samples=self._splits[split_key],
            data_dir=self.config.data_dir,
            views=self._view_specs(),
            resize_view=self._resize_view(),
            transform=transform,
            label_mode=data.label_mode,
            split_name_extension=data.split_name_extension,
            file_extension=data.file_extension,
        )

    # -- Lightning hooks -----------------------------------------------------
    def prepare_data(self) -> None:
        """No download/prep step: the raw NIfTI scans already live on disk."""

    def setup(self, stage: Optional[str] = None) -> None:
        """Read the split JSON and build the train/val/test datasets."""
        data = self.config.data
        raw_splits = read_split(self.config.split_path, data.split_view_key)

        # Resolve logical stages -> concrete split keys in the JSON.
        for logical in {self.config.splits.train, self.config.splits.val, self.config.splits.test}:
            if logical not in raw_splits:
                raise KeyError(
                    f"Split '{logical}' not present in {self.config.split_path} "
                    f"(available: {list(raw_splits)})."
                )
        self._splits = raw_splits

        num_frames = data.num_frames
        image_size = data.image_size
        train_transform = build_transform_pipeline(
            self.config.transforms.train, num_frames, image_size
        )
        eval_transform = build_transform_pipeline(
            self.config.transforms.val, num_frames, image_size
        )

        self._datasets = {
            "train": self._build_dataset(self.config.splits.train, train_transform),
            "val": self._build_dataset(self.config.splits.val, eval_transform),
            "test": self._build_dataset(self.config.splits.test, eval_transform),
        }

    # -- dataloaders ---------------------------------------------------------
    def _loader_kwargs(self, num_workers: int) -> dict:
        loader = self.config.loader
        kwargs = dict(
            num_workers=num_workers,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
        )
        # These options are only valid when using worker subprocesses.
        if num_workers > 0:
            kwargs["persistent_workers"] = loader.persistent_workers
            kwargs["prefetch_factor"] = loader.prefetch_factor
        return kwargs

    def train_dataloader(self) -> DataLoader:
        loader = self.config.loader
        dataset = self._datasets["train"]
        sampler = None
        shuffle = True
        if loader.use_weighted_sampler:
            # Class-balanced oversampling (mirrors builder.build_loader).
            sampler = WeightedRandomSampler(
                weights=dataset.sample_weights(),
                num_samples=len(dataset),
                replacement=True,
            )
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            **self._loader_kwargs(loader.num_workers),
        )

    def val_dataloader(self) -> DataLoader:
        loader = self.config.loader
        return DataLoader(
            self._datasets["val"],
            batch_size=loader.val_batch_size,
            shuffle=False,
            **self._loader_kwargs(loader.val_num_workers),
        )

    def test_dataloader(self) -> DataLoader:
        loader = self.config.loader
        return DataLoader(
            self._datasets["test"],
            batch_size=loader.val_batch_size,
            shuffle=False,
            **self._loader_kwargs(loader.val_num_workers),
        )


def _smoke_test() -> None:
    """Build the module and pull one batch from each split for inspection."""
    module = AmTICISDataModule()
    module.setup("fit")
    cfg = module.config
    print(f"active views : {cfg.views.active}")
    print(f"num_frames   : {cfg.data.num_frames}  image_size: {cfg.data.image_size}")
    print(f"label_mode   : {cfg.data.label_mode}  num_classes: {cfg.data.num_classes}")
    for split in ("train", "val", "test"):
        print(f"{split:5s} samples: {len(module._datasets[split])}")

    batch = next(iter(module.val_dataloader()))
    for view in cfg.views.active:
        print(f"val batch '{view}': {tuple(batch[view].shape)} dtype={batch[view].dtype}")
    print(f"val labels   : {tuple(batch['label'].shape)}")


if __name__ == "__main__":
    _smoke_test()
