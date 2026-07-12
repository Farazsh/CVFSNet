"""Shared data plumbing for MedGemma adaptation experiments.

The single source of truth for frame count, image size, and active views is the
experiment YAML. Nothing here hard-codes those values: they flow into the
repository's existing ``ResizeView`` trilinear resampling through
``build_pipeline_config``.

Two selection protocols are supported, chosen by ``selection.protocol``:

``internal`` (the default)
    The deterministic split already used by the zero-shot experiments: the 261
    ``train`` studies are cut 80/20 into development and tuning, so every
    MedGemma strategy selects checkpoints and thresholds on the same 53 studies
    and leaves the same 150 ``val`` studies locked and unread until
    ``--phase final``.

``full_train_val``
    Train on all 261 ``train`` studies and early-stop on the 150 ``val``
    studies, which is the protocol the VideoMAE and CVFSNet experiments already
    use. ``tuning`` therefore *aliases* ``final``: the run selects its epoch on
    the same studies it reports. That is an optimistic bias, and it is
    deliberate -- it is what makes these numbers comparable to the other model
    families. It is not an estimate of held-out performance.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from torch.utils.data import DataLoader, Subset

from amticis_pipeline.dataset_loader import AmTICISDataModule
from amticis_training.config import build_pipeline_config, deep_update

from .common import REPO_ROOT, assert_disjoint_studies, make_internal_split, write_split_manifest

PROTOCOLS = ("internal", "full_train_val")


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
        self.protocol = str(selection.get("protocol", "internal"))
        if self.protocol not in PROTOCOLS:
            raise ValueError(
                f"Unknown selection.protocol '{self.protocol}'; expected one of {PROTOCOLS}."
            )

        train_dataset = self.eval_module._datasets["train"]
        self.final: list[str] = list(self.eval_module._datasets["val"].samples)

        if self.protocol == "internal":
            splits = make_internal_split(
                train_dataset.samples,
                [train_dataset.label_at(index) for index in range(len(train_dataset))],
                tuning_fraction=float(selection["tuning_fraction"]),
                seed=int(selection["split_seed"]),
            )
            self.development: list[str] = splits["development"]
            self.tuning: list[str] = splits["tuning"]
        else:
            self.development = sorted(train_dataset.samples)
            self.tuning = list(self.final)

        manifest = {"development": self.development, "tuning": self.tuning, "final": self.final}
        # Under full_train_val, ``tuning`` is ``final`` by construction, so a pairwise
        # check over all three would flag those 150 studies as overlapping themselves.
        # What must hold either way is that nothing trained on is ever evaluated on.
        assert_disjoint_studies(
            manifest
            if self.protocol == "internal"
            else {"development": self.development, "final": self.final}
        )
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
        """Resolve labels for any study name, whichever dataset it lives in.

        Under ``full_train_val`` the tuning names are ``val`` studies, so looking
        them up in the ``train`` dataset alone raises ``KeyError``.
        """
        labels: dict[str, int] = {}
        for key in ("train", "val"):
            dataset = self.eval_module._datasets[key]
            for position, name in enumerate(dataset.samples):
                labels.setdefault(name, dataset.label_at(position))
        missing = [name for name in names if name not in labels]
        if missing:
            raise KeyError(f"{len(missing)} studies are in no dataset (e.g. {missing[0]}).")
        return [labels[name] for name in names]

    def loader(self, split: str, *, shuffle: bool, num_workers: int | None = None) -> DataLoader:
        """One study per batch, matching the plan's ``micro_batch_size: 1``."""
        if split == "development":
            module, key, names = self.train_module, "train", self.development
        elif split == "tuning":
            # The tuning studies are drawn from ``train`` under the internal protocol
            # and from ``val`` under full_train_val; either way they are scored
            # through the deterministic (un-augmented) module.
            tuning_key = "train" if self.protocol == "internal" else "val"
            module, key, names = self.eval_module, tuning_key, self.tuning
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
