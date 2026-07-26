"""AP DSA loading for DINOv3: three temporal frames become RGB-like channels."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.transforms.transforms import RandomResizedCrop

try:
    from lightning.pytorch import LightningDataModule
except Exception:  # pragma: no cover
    from pytorch_lightning import LightningDataModule

from amticis_pipeline.utils import (
    load_nifti,
    map_label,
    parse_label_token,
    read_split,
    resolve_view_filename,
)


class TemporalChannelTransform:
    """Apply one spatial transform to all three temporal channels."""

    def __init__(self, image_size: int, config: Mapping[str, Any]) -> None:
        self.image_size = int(image_size)
        self.hflip_p = float(config.get("horizontal_flip_probability", 0.0))
        self.rotation_p = float(config.get("rotation_probability", 0.0))
        self.rotation_degrees = float(config.get("rotation_degrees", 0.0))
        self.crop_scale = tuple(config.get("crop_scale", (1.0, 1.0)))
        self.crop_ratio = tuple(config.get("crop_ratio", (1.0, 1.0)))

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() < self.hflip_p:
            image = TF.hflip(image)
        if random.random() < self.rotation_p:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
        if self.crop_scale != (1.0, 1.0) or self.crop_ratio != (1.0, 1.0):
            top, left, height, width = RandomResizedCrop.get_params(
                image, scale=self.crop_scale, ratio=self.crop_ratio
            )
            image = TF.resized_crop(
                image,
                top,
                left,
                height,
                width,
                (self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        return image


class DINOv3TICIDataset(Dataset):
    """Load one AP NIfTI study as a normalized ``(3, H, W)`` tensor."""

    def __init__(
        self,
        samples: Sequence[str],
        data_dir: str | Path,
        config: Mapping[str, Any],
        image_mean: Sequence[float],
        image_std: Sequence[float],
        training: bool,
    ) -> None:
        self.samples = list(samples)
        self.data_dir = Path(data_dir)
        self.config = dict(config)
        self.num_frames = int(config["num_frames"])
        if self.num_frames != 3:
            raise ValueError("DINOv3 temporal-channel inputs require data.num_frames=3.")
        self.image_size = int(config["image_size"])
        self.mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        if self.mean.numel() != 3 or self.std.numel() != 3 or torch.any(self.std <= 0):
            raise ValueError("DINOv3 normalization requires three means and positive stds.")
        self.training = bool(training)
        self.augmentation = TemporalChannelTransform(
            self.image_size, config.get("augmentation", {})
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _path(self, base_name: str) -> Path:
        view = self.config["view"]
        name = resolve_view_filename(
            base_name,
            replace_from=str(view["replace_from"]),
            replace_to=str(view["replace_to"]),
            split_name_extension=str(self.config["split_name_extension"]),
            file_extension=str(self.config["file_extension"]),
        )
        return self.data_dir / name

    def _resize(self, scan_hwt) -> torch.Tensor:
        scan = torch.as_tensor(scan_hwt, dtype=torch.float32)
        if scan.ndim != 3 or not torch.isfinite(scan).all():
            raise ValueError(f"Expected a finite (H,W,T) scan, got {tuple(scan.shape)}.")
        volume = scan.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        interpolation = str(self.config.get("interpolation", "trilinear"))
        kwargs: dict[str, Any] = {"mode": interpolation}
        if interpolation in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = bool(self.config.get("align_corners", True))
        resized = F.interpolate(
            volume,
            size=(self.num_frames, self.image_size, self.image_size),
            **kwargs,
        )
        return resized[0, 0].contiguous()

    def _scale(self, image: torch.Tensor) -> torch.Tensor:
        lower = float(self.config.get("percentile_lower", 1.0)) / 100.0
        upper = float(self.config.get("percentile_upper", 99.0)) / 100.0
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")
        lo = torch.quantile(image, lower)
        hi = torch.quantile(image, upper)
        if not torch.isfinite(lo) or not torch.isfinite(hi) or hi <= lo:
            raise ValueError("Cannot normalize a non-finite or degenerate DSA scan.")
        return image.clamp(lo, hi).sub(lo).div(hi - lo)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name = self.samples[index]
        image = self._scale(self._resize(load_nifti(self._path(name))))
        if self.training:
            image = self.augmentation(image)
        pixel_values = image.sub(self.mean).div(self.std).contiguous()
        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(map_label(name, "binary"), dtype=torch.long),
            "name": name,
            "raw_grade": parse_label_token(name),
        }


class DINOv3DataModule(LightningDataModule):
    def __init__(
        self,
        config: Mapping[str, Any],
        image_mean: Sequence[float],
        image_std: Sequence[float],
        project_root: str | Path = ".",
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.project_root = Path(project_root).resolve()
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self.train_dataset: DINOv3TICIDataset | None = None
        self.val_dataset: DINOv3TICIDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        splits = read_split(
            self.project_root / self.config["split_file"],
            str(self.config["split_view_key"]),
        )
        common = dict(
            data_dir=self.project_root / self.config["root_dir"],
            config=self.config,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
        self.train_dataset = DINOv3TICIDataset(splits["train"], training=True, **common)
        self.val_dataset = DINOv3TICIDataset(splits["val"], training=False, **common)

    def _kwargs(self, workers: int) -> dict[str, Any]:
        loader = self.config["loader"]
        kwargs: dict[str, Any] = {
            "num_workers": workers,
            "pin_memory": bool(loader.get("pin_memory", False)),
        }
        if workers:
            kwargs["persistent_workers"] = bool(loader.get("persistent_workers", False))
            kwargs["prefetch_factor"] = int(loader.get("prefetch_factor", 2))
        return kwargs

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader.")
        loader = self.config["loader"]
        return DataLoader(
            self.train_dataset,
            batch_size=int(loader["batch_size"]),
            shuffle=True,
            drop_last=False,
            **self._kwargs(int(loader.get("num_workers", 0))),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader.")
        loader = self.config["loader"]
        return DataLoader(
            self.val_dataset,
            batch_size=int(loader.get("val_batch_size", loader["batch_size"])),
            shuffle=False,
            drop_last=False,
            **self._kwargs(int(loader.get("val_num_workers", 0))),
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
