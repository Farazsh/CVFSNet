"""Shared data plumbing for MedGemma adaptation experiments.

The single source of truth for frame count, image size, and active views is the
experiment YAML. Nothing here hard-codes those values: they flow into the
repository's existing ``ResizeView`` trilinear resampling through
``build_pipeline_config``.

The internal development/tuning split is the deterministic split already used by
the zero-shot experiments, so every MedGemma strategy selects checkpoints and
thresholds on the same 53 studies and leaves the same 150 studies locked.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from torch.utils.data import DataLoader, Subset

from amticis_pipeline.dataset_loader import AmTICISDataModule
from amticis_training.config import build_pipeline_config, deep_update

from .common import REPO_ROOT, assert_disjoint_studies, make_internal_split, write_split_manifest


def _pipeline_module(config: Mapping[str, Any], *, augment: bool) -> AmTICISDataModule:
    """Build a data module whose ``train`` dataset is augmented or deterministic."""
    data_cfg = json.loads(json.dumps(dict(config["data"])))  # deep copy of plain YAML data
    if not augment:
        deep_update(data_cfg, {"pipeline_overrides": {"transforms": {"train": []}}})

    module = AmTICISDataModule(config=build_pipeline_config({"data": data_cfg}))
    module.setup("fit")

    resolved = module.config
    expected_frames = int(config["input"]["num_frames"])
    expected_size = int(config["input"]["image_size"])
    expected_views = list(config["input"]["views"])
    if resolved.data.num_frames != expected_frames:
        raise ValueError(
            f"Pipeline num_frames {resolved.data.num_frames} != input.num_frames {expected_frames}."
        )
    if resolved.data.image_size != expected_size:
        raise ValueError(
            f"Pipeline image_size {resolved.data.image_size} != input.image_size {expected_size}."
        )
    if resolved.views.active != expected_views:
        raise ValueError(
            f"Pipeline views {resolved.views.active} != input.views {expected_views}."
        )
    if resolved.data.label_mode != "binary":
        raise ValueError("MedGemma binary TICI experiments require data.label_mode: binary.")
    return module


class MedGemmaData:
    """Development / tuning / final study sets over the shared AmTICIS pipeline.

    ``train_module`` applies the configured augmentation; ``eval_module`` is a
    second, deterministic view of the same studies used for tuning and final
    scoring so that model selection never sees a randomly augmented scan.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.train_module = _pipeline_module(config, augment=True)
        self.eval_module = _pipeline_module(config, augment=False)

        selection = config["selection"]
        train_dataset = self.eval_module._datasets["train"]
        splits = make_internal_split(
            train_dataset.samples,
            [train_dataset.label_at(index) for index in range(len(train_dataset))],
            tuning_fraction=float(selection["tuning_fraction"]),
            seed=int(selection["split_seed"]),
        )
        self.development: list[str] = splits["development"]
        self.tuning: list[str] = splits["tuning"]
        self.final: list[str] = list(self.eval_module._datasets["val"].samples)

        manifest = {"development": self.development, "tuning": self.tuning, "final": self.final}
        assert_disjoint_studies(manifest)
        self._reconcile_manifest(REPO_ROOT / str(selection["manifest_path"]), manifest)

    @staticmethod
    def _reconcile_manifest(path, manifest: Mapping[str, Sequence[str]]) -> None:
        """Fail loudly rather than silently re-splitting studies across experiments."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing != dict(manifest):
                raise RuntimeError(
                    f"Split manifest {path} disagrees with the deterministic split. "
                    "The zero-shot and adaptation experiments must share one split."
                )
        else:
            write_split_manifest(path, manifest)

    def labels_for(self, names: Sequence[str]) -> list[int]:
        dataset = self.eval_module._datasets["train"]
        index = {name: position for position, name in enumerate(dataset.samples)}
        return [dataset.label_at(index[name]) for name in names]

    def loader(self, split: str, *, shuffle: bool, num_workers: int | None = None) -> DataLoader:
        """One study per batch, matching the plan's ``micro_batch_size: 1``."""
        if split == "development":
            module, key, names = self.train_module, "train", self.development
        elif split == "tuning":
            module, key, names = self.eval_module, "train", self.tuning
        elif split == "final":
            module, key, names = self.eval_module, "val", self.final
        else:
            raise ValueError(f"Unknown split '{split}'.")

        dataset = module._datasets[key]
        wanted = set(names)
        indices = [i for i, name in enumerate(dataset.samples) if name in wanted]
        if len(indices) != len(wanted):
            raise ValueError(f"Split '{split}' is not fully present in the {key} dataset.")

        loader_cfg = module.config.loader
        workers = loader_cfg.num_workers if num_workers is None else num_workers
        kwargs: dict[str, Any] = {
            "batch_size": 1,
            "shuffle": shuffle,
            "num_workers": workers,
            "pin_memory": False,
            "drop_last": False,
        }
        if workers:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = loader_cfg.prefetch_factor
        return DataLoader(Subset(dataset, indices), **kwargs)
