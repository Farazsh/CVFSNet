"""Export a small, human-inspectable sample of the AmTICIS dataset.

Two processing modes are supported:

* ``default``    -- no processing. Just the existing NIfTI->image conversion:
                    every native frame is written at its original resolution
                    and contrast.
* ``CVFS_paper`` -- the processing used by the CVFSNet paper's data loader
                    (``Data/dataset.py::AmTICIS.__resize_view``): a single 3D
                    trilinear interpolation that resamples every sequence to
                    ``FAST_TIME_SIZE`` frames at ``VISUAL_SIZE x VISUAL_SIZE``.
                    This yields the 8 frames the network actually ingests.

For both modes the output keeps the same layout:
    <OUT_DIR>/<LABEL>/<SUBJECT_ID>/<AP|sagittal>/frame_XXX.png

The ``CVFS_paper`` mode reuses the exact same subjects/labels that were
exported by ``default`` (read from ``AmTICIS_processed``), so the two trees are
directly comparable frame-tree for frame-tree.
"""

import argparse
import os
import re

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = os.path.join(ROOT, "AmTICIS")

# Output directory per mode.
OUT_DIRS = {
    "default": os.path.join(ROOT, "AmTICIS_processed"),
    "CVFS_paper": os.path.join(ROOT, "AmTICIS_processed_CVFS_paper"),
}

# Number of subjects to export per label (used only when selecting fresh).
N_PER_LABEL = 3

# CVFSNet data-loader defaults (see config.py: DATA.*.DataPara).
FAST_TIME_SIZE = 8   # number of output frames after temporal interpolation
VISUAL_SIZE = 256    # spatial size the paper resamples to

# Map an output folder name -> the token used inside the file names.
# The dataset stores coronal (AP) views as "_C" and sagittal views as "_S".
LABELS = {
    "T0": "T0",
    "T1": "T1",
    "T2a": "T2A",
    "T2b": "T2B",
    "T3": "T3",
}


# --------------------------------------------------------------------------- #
# NIfTI loading + processing functions
# --------------------------------------------------------------------------- #
def load_raw(path: str) -> np.ndarray:
    """Load a NIfTI DSA sequence as a float32 ``(H, W, T)`` array.

    Values are already scaled to roughly 0-255, so nothing is rescaled here;
    contrast is preserved for the downstream processing functions.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)


def process_default(raw: np.ndarray) -> np.ndarray:
    """No processing: the plain NIfTI->image conversion.

    Returns ``(T, H, W)`` uint8 frames at the original resolution and contrast
    (clip to the display range and cast, no normalization, no resampling).
    """
    frames = np.transpose(raw, (2, 0, 1))  # (H, W, T) -> (T, H, W)
    return np.clip(frames, 0, 255).astype(np.uint8)


def process_cvfs_paper(
    raw: np.ndarray,
    t: int = FAST_TIME_SIZE,
    visual: int = VISUAL_SIZE,
) -> np.ndarray:
    """Reproduce ``AmTICIS.__resize_view`` from the CVFSNet data loader.

    A single 3D trilinear interpolation resamples the ``(H, W, T)`` volume to
    ``(t, visual, visual)`` -- i.e. the temporal axis is interpolated down to
    ``t`` frames and the spatial axes to ``visual x visual`` at the same time.
    Returns ``(t, visual, visual)`` uint8 frames (contrast preserved; the
    per-clip Z-normalization applied later in training is intentionally skipped
    so the frames remain directly viewable).
    """
    view = (
        torch.tensor(raw, dtype=torch.float32)
        .permute(2, 0, 1)  # (H, W, T) -> (T, H, W)
        .unsqueeze(0)
        .unsqueeze(0)  # -> (1, 1, T, H, W)
    )
    rv = F.interpolate(
        view,
        (t, visual, visual),
        mode="trilinear",
        align_corners=True,
    ).contiguous()[0]  # -> (1, t, visual, visual)
    frames = rv[0].numpy()  # -> (t, visual, visual)
    return np.clip(frames, 0, 255).astype(np.uint8)


PROCESSORS = {
    "default": process_default,
    "CVFS_paper": process_cvfs_paper,
}


# --------------------------------------------------------------------------- #
# Subject selection
# --------------------------------------------------------------------------- #
def subject_id(filename: str) -> str:
    """Return the 4-digit subject id that prefixes every file name."""
    return re.match(r"(\d+)_", filename).group(1)


def is_coronal(name: str) -> bool:
    """True if the file name carries the coronal (AP) view marker ``_C``."""
    return re.search(r"_C(?:#|_)", name) is not None


def coronal_files_for(token: str):
    """All coronal (AP) files for a given label token, sorted by subject id."""
    files = []
    for name in os.listdir(SRC_DIR):
        if not name.endswith(".nii.gz"):
            continue
        if "_%s_" % token not in name and "_%s_" % token.upper() not in name:
            continue
        if not is_coronal(name):
            continue
        files.append(name)
    return sorted(files, key=subject_id)


def find_coronal_for_subject(sid: str):
    """Locate the coronal source file for a given subject id."""
    for name in sorted(os.listdir(SRC_DIR)):
        if (
            name.startswith(sid + "_")
            and name.endswith(".nii.gz")
            and is_coronal(name)
        ):
            return name
    return None


def select_fresh():
    """Pick the first ``N_PER_LABEL`` subjects per label that have both views.

    Returns a list of ``(label, subject_id, coronal_filename)`` tuples.
    """
    selection = []
    for label, token in LABELS.items():
        exported = 0
        for cor_name in coronal_files_for(token):
            if exported >= N_PER_LABEL:
                break
            if not os.path.exists(
                os.path.join(SRC_DIR, cor_name.replace("_C", "_S"))
            ):
                continue
            selection.append((label, subject_id(cor_name), cor_name))
            exported += 1
        if exported < N_PER_LABEL:
            print("  [warn] only found %d/%d for %s" % (exported, N_PER_LABEL, label))
    return selection


def select_from_existing(existing_dir: str):
    """Reuse the exact labels/subjects already exported under ``existing_dir``.

    Returns a list of ``(label, subject_id, coronal_filename)`` tuples.
    """
    selection = []
    for label in LABELS:
        label_dir = os.path.join(existing_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for sid in sorted(os.listdir(label_dir)):
            if not os.path.isdir(os.path.join(label_dir, sid)):
                continue
            cor_name = find_coronal_for_subject(sid)
            if cor_name is None:
                print("  [skip] no source NIfTI found for subject %s" % sid)
                continue
            selection.append((label, sid, cor_name))
    return selection


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def save_frames(frames: np.ndarray, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    n = frames.shape[0]
    width = max(3, len(str(n - 1)))
    for i in range(n):
        Image.fromarray(frames[i], mode="L").save(
            os.path.join(out_dir, "frame_%s.png" % str(i).zfill(width))
        )
    return n


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_mode(mode: str):
    processor = PROCESSORS[mode]
    out_dir = OUT_DIRS[mode]

    # CVFS_paper reuses exactly the subjects/labels exported by default so the
    # two trees line up one-to-one; default selects fresh.
    if mode == "CVFS_paper" and os.path.isdir(OUT_DIRS["default"]):
        selection = select_from_existing(OUT_DIRS["default"])
        print(
            "Reusing %d subjects from %s"
            % (len(selection), os.path.relpath(OUT_DIRS["default"], ROOT))
        )
    else:
        selection = select_fresh()

    os.makedirs(out_dir, exist_ok=True)
    print(">>> mode=%s -> %s" % (mode, os.path.relpath(out_dir, ROOT)))

    for label, sid, cor_name in selection:
        sag_name = cor_name.replace("_C", "_S")
        cor_path = os.path.join(SRC_DIR, cor_name)
        sag_path = os.path.join(SRC_DIR, sag_name)
        if not os.path.exists(sag_path):
            print("  [skip] no sagittal pair for %s" % cor_name)
            continue

        cor_frames = processor(load_raw(cor_path))
        sag_frames = processor(load_raw(sag_path))

        n_ap = save_frames(cor_frames, os.path.join(out_dir, label, sid, "AP"))
        n_sag = save_frames(
            sag_frames, os.path.join(out_dir, label, sid, "sagittal")
        )

        print(
            "%-4s subject %s | AP: %3d frames %s | sagittal: %3d frames %s"
            % (
                label,
                sid,
                n_ap,
                tuple(cor_frames.shape[1:]),
                n_sag,
                tuple(sag_frames.shape[1:]),
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=list(PROCESSORS.keys()) + ["both"],
        default="CVFS_paper",
        help="Which processing to apply (default: CVFS_paper).",
    )
    args = parser.parse_args()

    modes = ["default", "CVFS_paper"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_mode(mode)


if __name__ == "__main__":
    main()
