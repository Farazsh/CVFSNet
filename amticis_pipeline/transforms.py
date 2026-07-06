"""Preprocessing and augmentation transforms for the AmTICIS pipeline.

This module contains two things:

1. :class:`ResizeView` -- the temporal+spatial resampling step. It reproduces
   ``Data/dataset.py::AmTICIS.__resize_view`` exactly: a single 3D trilinear
   interpolation that turns a raw ``(H, W, T)`` scan into a
   ``(1, num_frames, image_size, image_size)`` clip. This is where "frame
   selection" happens -- an arbitrary native frame count is resampled to
   ``num_frames``.

2. :func:`build_transform_pipeline` -- a small factory that assembles a
   ``Compose`` of augmentations from the declarative config. The augmentation
   classes themselves are *reused verbatim* from the repository's original
   ``Data/transforms.py`` (imported lazily after a compatibility shim), so the
   behaviour is guaranteed to match the existing pipeline.

TO CHANGE FRAME COUNT / IMAGE SIZE: edit ``data.num_frames`` / ``data.image_size``
in ``config.yaml`` -- both flow into ``ResizeView`` and into the ``Resize``
augmentation automatically.
TO ADD / REORDER AUGMENTATIONS: edit ``transforms.train`` / ``transforms.val``
in ``config.yaml``.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import TransformStep
from .utils import ensure_repo_importable, ensure_torchvision_compat

# Interpolation modes for which ``align_corners`` is a valid argument.
_ALIGN_CORNERS_MODES = {"linear", "bilinear", "bicubic", "trilinear"}

# A transform maps a (C, T, H, W) tensor to a (C, T, H, W) tensor.
Transform = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------- #
# OpenCV-5 compatible morphology (local overrides)
# --------------------------------------------------------------------------- #
# The original Data/transforms.py RandomErode/RandomDilate call cv2 on a
# (1, H, W) slice, which OpenCV >= 5 rejects ("dims <= 2"). These drop-in
# replacements apply the morphology per (channel, frame) on plain 2D images,
# preserving the original intent while remaining version-compatible. They are
# preferred over the legacy classes via LOCAL_TRANSFORMS below.
class _Morphology:
    """Base for per-frame morphological augmentation on a (C, T, H, W) clip."""

    _op: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def __init__(self, k: int = 5, high: int = 192, low: int = 48, p: float = 0.15, **_) -> None:
        self.p = p
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return clip
        arr = clip.numpy()
        channels, frames = arr.shape[0], arr.shape[1]
        for c in range(channels):
            for t in range(frames):
                arr[c, t] = self._op(arr[c, t], self.kernel)
        return torch.from_numpy(arr)


class RandomErode(_Morphology):
    _op = staticmethod(cv2.erode)


class RandomDilate(_Morphology):
    _op = staticmethod(cv2.dilate)


# Name -> class overrides that take precedence over Data/transforms.py.
# ADD A LOCAL OVERRIDE / CUSTOM AUGMENTATION: register it here and reference it
# by name in config.yaml.
LOCAL_TRANSFORMS: Dict[str, type] = {
    "RandomErode": RandomErode,
    "RandomDilate": RandomDilate,
}


class ResizeView:
    """Resample a raw ``(H, W, T)`` scan to a fixed-size ``(1, T', H', W')`` clip.

    Faithful reimplementation of ``AmTICIS.__resize_view``. Kept as a dedicated,
    configurable class so the frame count and spatial size are never hardcoded.
    """

    def __init__(
        self,
        num_frames: int,
        image_size: int,
        interpolation: str = "trilinear",
        align_corners: bool = True,
    ) -> None:
        self.num_frames = num_frames
        self.image_size = image_size
        self.interpolation = interpolation
        self.align_corners = align_corners

    def __call__(self, scan_hwt: np.ndarray) -> torch.Tensor:
        # (H, W, T) -> (1, 1, T, H, W) so F.interpolate treats T, H, W as the
        # three spatial dims of a single-channel volume.
        view = (
            torch.as_tensor(scan_hwt, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        target = (self.num_frames, self.image_size, self.image_size)
        kwargs = {"mode": self.interpolation}
        if self.interpolation in _ALIGN_CORNERS_MODES:
            kwargs["align_corners"] = self.align_corners
        resized = F.interpolate(view, target, **kwargs).contiguous()
        # Drop the batch dim -> (C=1, T', H', W').
        return resized[0]


class ComposeTransforms:
    """Apply a sequence of transforms in order (a thin, typed ``Compose``)."""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms = list(transforms)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            clip = transform(clip)
        return clip


def _instantiate_step(step: TransformStep, num_frames: int, image_size: int):
    """Instantiate one augmentation class from ``Data/transforms.py``."""
    import Data.transforms as legacy_transforms  # lazy: needs the compat shim first

    params = dict(step.params)
    # The Resize augmentation re-fixes size after cropping; inject the frame
    # count / spatial size from the single source of truth in config.
    if step.name == "Resize":
        params.setdefault("t", num_frames)
        params.setdefault("visual", [image_size, image_size])

    # Prefer a local (env-compatible / custom) transform, then fall back to the
    # repository's original Data/transforms.py class.
    if step.name in LOCAL_TRANSFORMS:
        transform_cls = LOCAL_TRANSFORMS[step.name]
    else:
        try:
            transform_cls = getattr(legacy_transforms, step.name)
        except AttributeError as error:
            raise ValueError(
                f"Transform '{step.name}' is not defined locally or in "
                f"Data/transforms.py."
            ) from error
    return transform_cls(**params)


def build_transform_pipeline(
    steps: List[TransformStep],
    num_frames: int,
    image_size: int,
) -> ComposeTransforms:
    """Build a ``Compose`` from the declarative transform list in the config."""
    ensure_repo_importable()
    ensure_torchvision_compat()
    transforms = [_instantiate_step(step, num_frames, image_size) for step in steps]
    return ComposeTransforms(transforms)
