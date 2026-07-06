"""Typed configuration objects for the AmTICIS data pipeline.

The configuration is authored in ``config.yaml`` and parsed here into small,
readable dataclasses. Keeping the schema in one place means the rest of the
code can rely on attribute access (with type hints and IDE completion) instead
of digging through nested dictionaries.

To add a new option:
    1. add it to ``config.yaml``;
    2. add a typed field to the matching dataclass below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Number of classes produced by each labelling mode. Referenced by models /
# losses that need to know the classifier width.
NUM_CLASSES_BY_MODE: Dict[str, int] = {"full": 5, "fuse01": 4, "binary": 2}


@dataclass
class DataConfig:
    root_dir: str
    split_file: str
    split_view_key: str
    split_name_extension: str
    file_extension: str
    label_mode: str
    num_frames: int
    image_size: int
    interpolation: str
    align_corners: bool

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES_BY_MODE[self.label_mode]


@dataclass
class ViewDefinition:
    """Rule for deriving one view's file name from the coronal base name."""

    replace_from: str
    replace_to: str


@dataclass
class ViewsConfig:
    active: List[str]
    definitions: Dict[str, ViewDefinition]

    def __post_init__(self) -> None:
        unknown = set(self.active) - set(self.definitions)
        if unknown:
            raise ValueError(
                f"Active views {sorted(unknown)} have no entry under "
                f"views.definitions (known: {sorted(self.definitions)})."
            )


@dataclass
class SplitsConfig:
    train: str
    val: str
    test: str


@dataclass
class LoaderConfig:
    batch_size: int
    num_workers: int
    val_batch_size: int
    val_num_workers: int
    pin_memory: bool
    drop_last: bool
    use_weighted_sampler: bool
    persistent_workers: bool
    prefetch_factor: int


@dataclass
class TransformStep:
    """A single declarative transform entry: a class name plus kwargs."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformsConfig:
    train: List[TransformStep]
    val: List[TransformStep]


@dataclass
class PipelineConfig:
    data: DataConfig
    views: ViewsConfig
    splits: SplitsConfig
    loader: LoaderConfig
    transforms: TransformsConfig
    # Absolute directory that relative paths (root_dir, split_file) resolve
    # against. Defaults to the repository root.
    project_root: Path = field(default_factory=Path.cwd)

    # -- convenience resolved paths ------------------------------------------
    @property
    def data_dir(self) -> Path:
        return (self.project_root / self.data.root_dir).resolve()

    @property
    def split_path(self) -> Path:
        return (self.project_root / self.data.split_file).resolve()


def _parse_transform_steps(raw: List[Dict[str, Any]]) -> List[TransformStep]:
    return [
        TransformStep(name=step["name"], params=step.get("params", {}) or {})
        for step in raw
    ]


def load_config(
    config_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> PipelineConfig:
    """Load ``config.yaml`` into a :class:`PipelineConfig`.

    Args:
        config_path: Path to the YAML file. Defaults to the ``config.yaml`` that
            sits next to this module.
        project_root: Directory that relative data paths resolve against.
            Defaults to the current working directory (the repo root when the
            pipeline is run from there).
    """
    config_path = Path(config_path) if config_path else Path(__file__).with_name("config.yaml")
    with open(config_path, "r") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle)

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
        project_root=Path(project_root).resolve() if project_root else Path.cwd(),
    )
