"""Helper functions shared across the AmTICIS data pipeline.

Responsibilities kept here (each is a small, independently testable unit):
    * making the repository importable + patching a torchvision incompatibility
      so the original ``Data/transforms.py`` classes can be reused verbatim;
    * reading the train/val split JSON;
    * mapping a scan file name to its integer mTICI label;
    * deriving a view's file path from the coronal (base) file name;
    * loading a raw NIfTI scan into a numpy array.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk

# Repository root = two levels up from this file (…/CVFSNet/amticis_pipeline/utils.py).
REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_repo_importable() -> None:
    """Put the repository root on ``sys.path`` so ``import Data`` works."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_torchvision_compat() -> None:
    """Shim the ``torchvision.transforms.functional_tensor`` module.

    ``Data/transforms.py`` imports ``torchvision.transforms.functional_tensor``,
    which torchvision removed in 0.17+. That symbol is only used by translate
    helpers which are *not* part of the train/val pipeline, so we alias it to
    the still-present ``torchvision.transforms.functional`` to let the module
    import cleanly. This patches only the in-memory module table; no files are
    modified.
    """
    name = "torchvision.transforms.functional_tensor"
    if name in sys.modules:
        return
    try:
        import torchvision.transforms.functional as F  # noqa: WPS433 (local import by design)
    except Exception:  # pragma: no cover - torchvision always present in this env
        return
    shim = types.ModuleType(name)
    shim.__dict__.update(F.__dict__)
    sys.modules[name] = shim


def read_split(split_path: Path, view_key: str) -> Dict[str, List[str]]:
    """Return ``{split_name: [coronal_file_name, ...]}`` from the split JSON."""
    with open(split_path, "r") as handle:
        payload = json.load(handle)
    if view_key not in payload:
        raise KeyError(
            f"Split key '{view_key}' not found in {split_path} "
            f"(available: {list(payload)})."
        )
    return payload[view_key]


# --------------------------------------------------------------------------- #
# Label mapping (mirrors Data/dataset.py::AmTICIS.__map_label)
# --------------------------------------------------------------------------- #
_FULL_LABELS = {"T0": 0, "T1": 1, "T2A": 2, "T2B": 3, "T3": 4}
_FUSE01_LABELS = {"T0": 0, "T1": 0, "T2A": 1, "T2B": 2, "T3": 3}
_BINARY_LABELS = {"T0": 0, "T1": 0, "T2A": 0, "T2B": 1, "T3": 1}

_LABEL_TABLES = {
    "full": _FULL_LABELS,
    "fuse01": _FUSE01_LABELS,
    "binary": _BINARY_LABELS,
}


def parse_label_token(file_name: str) -> str:
    """Extract the raw mTICI token (e.g. ``T2A``) from a scan file name.

    Matches the original ``file_dir.split('_')[1]`` logic and upper-cases so
    that ``T2a``/``T2A`` are treated identically.
    """
    return Path(file_name).name.split("_")[1].upper()


def map_label(file_name: str, label_mode: str) -> int:
    """Map a scan file name to its integer class under ``label_mode``."""
    table = _LABEL_TABLES.get(label_mode)
    if table is None:
        raise ValueError(
            f"Unknown label_mode '{label_mode}' (expected one of {list(_LABEL_TABLES)})."
        )
    token = parse_label_token(file_name)
    if token not in table:
        raise ValueError(f"Unsupported label token '{token}' from '{file_name}'.")
    return table[token]


# --------------------------------------------------------------------------- #
# View / path resolution
# --------------------------------------------------------------------------- #
def resolve_view_filename(
    base_name: str,
    replace_from: str,
    replace_to: str,
    split_name_extension: str,
    file_extension: str,
) -> str:
    """Turn a split-JSON entry into the on-disk file name for one view.

    Example: ``0180_T3_C_anon.dcm`` --(sagittal: _C -> _S)--> ``0180_T3_S_anon.nii.gz``.
    """
    name = base_name.replace(split_name_extension, file_extension)
    if replace_from != replace_to:
        name = name.replace(replace_from, replace_to)
    return name


def load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI scan as a float32 ``(H, W, T)`` array (contrast preserved)."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
