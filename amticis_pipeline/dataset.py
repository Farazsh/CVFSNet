"""``AmTICISDataset`` -- a torch ``Dataset`` over raw AmTICIS NIfTI scans.

Given a list of coronal (base) file names, the dataset:
    1. maps each name to its integer mTICI label;
    2. for every *active* view, loads the raw ``(H, W, T)`` NIfTI, resamples it
       to a fixed ``(1, num_frames, image_size, image_size)`` clip
       (``ResizeView``), then applies the configured transform pipeline;
    3. returns one dict per sample.

The class is deliberately decoupled from the config object: it receives only
the concrete pieces it needs, which makes it trivial to reuse for a different
dataset, view set, or transform stack.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .transforms import ComposeTransforms, ResizeView
from .utils import load_nifti, map_label, resolve_view_filename


@dataclass(frozen=True)
class ViewSpec:
    """A named rule for locating one view's file from the coronal base name.

    ADD A NEW VIEW: create another ``ViewSpec`` (usually built from config) and
    include it in ``views`` -- the dataset loops over whatever it is given.
    """

    name: str
    replace_from: str
    replace_to: str


class AmTICISDataset(Dataset):
    """Dataset yielding per-view clips + label for each AmTICIS study."""

    def __init__(
        self,
        samples: List[str],
        data_dir: Path,
        views: List[ViewSpec],
        resize_view: ResizeView,
        transform: ComposeTransforms,
        label_mode: str,
        split_name_extension: str,
        file_extension: str,
    ) -> None:
        if not views:
            raise ValueError("At least one view must be active.")
        self.samples = samples
        self.data_dir = Path(data_dir)
        self.views = views
        self.resize_view = resize_view
        self.transform = transform
        self.label_mode = label_mode
        self.split_name_extension = split_name_extension
        self.file_extension = file_extension

    def __len__(self) -> int:
        return len(self.samples)

    def label_at(self, index: int) -> int:
        """Integer class label for a sample (used by the sampler)."""
        return map_label(self.samples[index], self.label_mode)

    def _view_path(self, base_name: str, view: ViewSpec) -> Path:
        file_name = resolve_view_filename(
            base_name,
            replace_from=view.replace_from,
            replace_to=view.replace_to,
            split_name_extension=self.split_name_extension,
            file_extension=self.file_extension,
        )
        return self.data_dir / file_name

    def _load_view(self, base_name: str, view: ViewSpec) -> torch.Tensor:
        scan = load_nifti(self._view_path(base_name, view))  # (H, W, T)
        clip = self.resize_view(scan)                        # (1, T', H', W')
        return self.transform(clip).contiguous()

    def __getitem__(self, index: int) -> Dict[str, object]:
        base_name = self.samples[index]
        sample: Dict[str, object] = {
            view.name: self._load_view(base_name, view) for view in self.views
        }
        # Shape (1,) long tensor, matching the original dataloader's label form.
        sample["label"] = torch.tensor(
            [map_label(base_name, self.label_mode)], dtype=torch.long
        )
        sample["name"] = base_name
        return sample

    def sample_weights(self) -> List[float]:
        """Inverse-frequency weight per sample for class-balanced sampling.

        Equivalent to ``AmTICIS.get_weighted_count`` (inverse class frequency),
        implemented with a ``Counter`` to avoid the original's fragile
        list-compaction indexing.
        """
        counts = Counter(self.label_at(i) for i in range(len(self)))
        total = len(self)
        return [total / counts[self.label_at(i)] for i in range(len(self))]
