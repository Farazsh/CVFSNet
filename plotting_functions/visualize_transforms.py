"""Visualize every training-time data augmentation used by the CVFSNet pipeline.

The CVFSNet data loader (``Data/dataset.py::AmTICIS.train_trans``) applies the
following transforms, in order, to each 8x256x256 clip:

    RandomErode, RandomDilate, TioClamp, TioRandomFlip, TioRandomAnisotropy,
    TioRandomMotion, TioRandomGhosting, TioRandomSpike, TioRandomBiasField,
    TioRandomBlur, TioRandomNoise, TioRandomGamma, RandomRotation, Crop,
    Resize, TioZNormalization

This script reproduces each of those transforms individually (same libraries
and same value ranges as ``Data/transforms.py``). Each clip is loaded straight
from the source NIfTI in ``AmTICIS`` and resampled to 8x256x256 with the exact
trilinear ``AmTICIS.__resize_view`` step from the data loader (no intermediate
PNGs), then each transform is applied and the result is written as:

    <OUT_DIR>/<DATA_TRANSFORM>/<LABEL>/<SUBJECT_ID>/<AP|sagittal>/frame_XXX.png

Each leaf folder also gets a ``params.txt`` describing the concrete random
values that were drawn (within the data loader's ranges) for that clip.

The subjects/labels processed are exactly those present under
``AmTICIS_processed_CVFS_paper`` (used only to enumerate which subjects to run).

Notes
-----
* ``Data/transforms.py`` is NOT imported directly: its top-level
  ``import torchvision.transforms.functional_tensor`` was removed in
  torchvision >= 0.17, so importing it would fail in this environment. The
  transforms are reconstructed here with identical parameters/ranges.
* Random values differ per (transform, subject, view) but are reproducible via
  a deterministic per-leaf seed (recorded in params.txt).
"""

import argparse
import hashlib
import math
import os
import re
import warnings

import cv2
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.realpath(__file__))
NIFTI_DIR = os.path.join(ROOT, "AmTICIS")
INDEX_DIR = os.path.join(ROOT, "AmTICIS_processed_CVFS_paper")
OUT_DIR = os.path.join(ROOT, "AmTICIS_transforms")

# CVFSNet data-loader defaults (see config.py: DATA.*.DataPara).
FAST_TIME_SIZE = 8
VISUAL_SIZE = 256

LABELS = ["T0", "T1", "T2a", "T2b", "T3"]
# View folder name -> NIfTI view marker ("_C" = coronal/AP, "_S" = sagittal).
VIEWS = {"AP": "_C", "sagittal": "_S"}


# --------------------------------------------------------------------------- #
# NIfTI loading (mirrors AmTICIS.__load_arrays / __resize_view)
# --------------------------------------------------------------------------- #
def is_coronal(name: str) -> bool:
    return re.search(r"_C(?:#|_)", name) is not None


def find_coronal_for_subject(sid: str):
    """Locate the coronal (AP) source NIfTI for a given subject id."""
    for name in sorted(os.listdir(NIFTI_DIR)):
        if name.startswith(sid + "_") and name.endswith(".nii.gz") and is_coronal(name):
            return name
    return None


def load_raw(path: str) -> np.ndarray:
    """Load a NIfTI DSA sequence as a float32 (H, W, T) array (contrast intact)."""
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)


def resize_view(raw: np.ndarray, t=FAST_TIME_SIZE, visual=VISUAL_SIZE) -> torch.Tensor:
    """Exact reproduction of AmTICIS.__resize_view: 3D trilinear -> (1, t, visual, visual)."""
    view = (
        torch.tensor(raw, dtype=torch.float32)
        .permute(2, 0, 1)  # (H, W, T) -> (T, H, W)
        .unsqueeze(0)
        .unsqueeze(0)  # -> (1, 1, T, H, W)
    )
    return F.interpolate(
        view, (t, visual, visual), mode="trilinear", align_corners=True
    ).contiguous()[0]  # -> (1, t, visual, visual) == CDHW


def load_clip(sid: str, view: str):
    """Load a clip for (subject, view) from NIfTI and resample it as the loader does.

    Returns (clip CDHW float, meta dict).
    """
    cor_name = find_coronal_for_subject(sid)
    if cor_name is None:
        raise FileNotFoundError("no coronal NIfTI for subject %s" % sid)
    fname = cor_name if VIEWS[view] == "_C" else cor_name.replace("_C", "_S")
    raw = load_raw(os.path.join(NIFTI_DIR, fname))  # (H, W, T)
    clip = resize_view(raw)  # (1, 8, 256, 256)
    meta = {
        "source_nifti": fname,
        "native_shape_HWT": tuple(raw.shape),
        "resampled_to": (FAST_TIME_SIZE, VISUAL_SIZE, VISUAL_SIZE),
    }
    return clip, meta


def leaf_seed(*parts) -> int:
    """Deterministic 31-bit seed from string parts (stable across processes)."""
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def save_clip(clip: torch.Tensor, out_dir: str, params: dict, transform: str):
    """Save a (C=1, D, H, W) float clip as PNGs + a params.txt in the leaf folder."""
    os.makedirs(out_dir, exist_ok=True)
    frames = np.clip(clip[0].numpy(), 0, 255).astype(np.uint8)  # (D, H, W)
    n = frames.shape[0]
    width = max(3, len(str(n - 1)))
    for i in range(n):
        Image.fromarray(frames[i], mode="L").save(
            os.path.join(out_dir, "frame_%s.png" % str(i).zfill(width))
        )
    with open(os.path.join(out_dir, "params.txt"), "w") as fh:
        fh.write("transform    : %s\n" % transform)
        fh.write("output_frames: %d x %s\n" % (n, (frames.shape[1], frames.shape[2])))
        fh.write("-" * 60 + "\n")
        for k, v in params.items():
            fh.write("%-18s: %s\n" % (k, v))


# --------------------------------------------------------------------------- #
# TorchIO application + history formatting
# --------------------------------------------------------------------------- #
def apply_tio(clip: torch.Tensor, transform) -> "tuple[torch.Tensor, str]":
    """Apply a torchio transform to a CDHW clip; return (out_clip, history_str).

    Mirrors the data loader's ``permute(0, 2, 3, 1)`` (CDHW -> C,H,W,D) before
    handing the tensor to torchio, then permutes back.
    """
    chwd = clip.permute(0, 2, 3, 1).contiguous()  # C, H, W, D
    subject = tio.Subject(image=tio.ScalarImage(tensor=chwd))
    out = transform(subject)
    out_clip = out.image.data.permute(0, 3, 1, 2).contiguous()  # back to CDHW
    return out_clip, format_history(out.history)


def _clean(v):
    if isinstance(v, dict) and "image" in v:
        v = v["image"]
    if callable(v):
        return "<fn>"
    if isinstance(v, np.ndarray):
        return np.array2string(np.round(v, 3), threshold=40, max_line_width=100)
    if isinstance(v, float):
        return round(v, 4)
    return v


def format_history(history) -> str:
    parts = []
    for h in history:
        args = {a: _clean(getattr(h, a, None)) for a in getattr(h, "args_names", [])}
        parts.append("%s(%s)" % (type(h).__name__, args))
    return " -> ".join(parts) if parts else "(none)"


# --------------------------------------------------------------------------- #
# Individual transforms (same ranges as Data/transforms.py)
# --------------------------------------------------------------------------- #
def tf_erode(clip, rng, seed):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    arr = clip.numpy().copy()
    for i in range(arr.shape[1]):  # per frame, erode the 2D (H, W) image
        arr[0, i] = cv2.erode(arr[0, i], kernel)
    return torch.from_numpy(arr), {
        "operation": "erode",
        "kernel": "MORPH_RECT (3,3)",
        "dataloader_p": 0.15,
    }


def tf_dilate(clip, rng, seed):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    arr = clip.numpy().copy()
    for i in range(arr.shape[1]):  # per frame, dilate the 2D (H, W) image
        arr[0, i] = cv2.dilate(arr[0, i], kernel)
    return torch.from_numpy(arr), {
        "operation": "dilate",
        "kernel": "MORPH_RECT (3,3)",
        "dataloader_p": 0.15,
    }


def tf_clamp(clip, rng, seed):
    out, hist = apply_tio(clip, tio.Clamp(16, 192))
    return out, {
        "out_min": 16,
        "out_max": 192,
        "dataloader_p": "0 (DISABLED in training)",
        "history": hist,
    }


def tf_flip(clip, rng, seed):
    torch.manual_seed(seed)
    out, hist = apply_tio(clip, tio.RandomFlip(axes=("A",), flip_probability=1.0))
    return out, {"axes": "('A',)", "flip_probability": 1.0, "history": hist}


def tf_anisotropy(clip, rng, seed):
    factor = rng.uniform(5, 10)
    axis = rng.choice([0, 1, 2])
    torch.manual_seed(seed)
    out, hist = apply_tio(
        clip, tio.RandomAnisotropy(axes=(axis,), downsampling=(factor, factor))
    )
    return out, {
        "axis": axis,
        "downsampling_factor": round(factor, 3),
        "range": "(5, 10)",
        "history": hist,
    }


def tf_motion(clip, rng, seed):
    torch.manual_seed(seed)
    out, hist = apply_tio(
        clip, tio.RandomMotion(degrees=60, translation=20, num_transforms=10)
    )
    return out, {
        "degrees_max": 60,
        "translation_max": 20,
        "num_transforms": 10,
        "seed": seed,
        "history": hist,
    }


def tf_ghosting(clip, rng, seed):
    num = rng.randint(3, 5)
    axis = rng.choice([0, 1, 2])
    torch.manual_seed(seed)
    out, hist = apply_tio(clip, tio.RandomGhosting(num_ghosts=(num, num), axes=(axis,)))
    return out, {
        "num_ghosts": num,
        "axis": axis,
        "num_ghosts_range": "(3, 5)",
        "history": hist,
    }


def tf_spike(clip, rng, seed):
    torch.manual_seed(seed)
    out, hist = apply_tio(clip, tio.RandomSpike(num_spikes=3, intensity=(1, 2)))
    return out, {
        "num_spikes": 3,
        "intensity_range": "(1, 2)",
        "seed": seed,
        "history": hist,
    }


def tf_biasfield(clip, rng, seed):
    torch.manual_seed(seed)
    out, hist = apply_tio(clip, tio.RandomBiasField(coefficients=0.3, order=2))
    return out, {
        "coefficients_max": 0.3,
        "order": 2,
        "seed": seed,
        "history": hist,
    }


def tf_blur(clip, rng, seed):
    std = rng.uniform(0, 3)
    out, hist = apply_tio(clip, tio.RandomBlur(std=(std, std)))
    return out, {"std": round(std, 3), "std_range": "(0, 3)", "history": hist}


def tf_noise(clip, rng, seed):
    std = rng.uniform(0, 0.5)
    torch.manual_seed(seed)
    out, hist = apply_tio(clip, tio.RandomNoise(mean=0, std=(std, std)))
    return out, {
        "std": round(std, 4),
        "mean": 0,
        "std_range": "(0, 0.5)",
        "note": "std is tiny vs the 0-255 intensity range (as in the paper)",
        "history": hist,
    }


def tf_gamma(clip, rng, seed):
    log_gamma = rng.uniform(-0.5, 0.5)
    out, hist = apply_tio(clip, tio.RandomGamma(log_gamma=(log_gamma, log_gamma)))
    return out, {
        "log_gamma": round(log_gamma, 4),
        "gamma": round(math.exp(log_gamma), 4),
        "log_gamma_range": "(-0.5, 0.5)",
        "history": hist,
    }


def tf_rotation(clip, rng, seed):
    angle = rng.uniform(-30, 30)
    frames = []
    for i in range(clip.shape[1]):
        fr = TF.rotate(
            clip[:, i],
            angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=[192.0],
        )
        frames.append(fr)
    out = torch.stack(frames, dim=1)  # (C, D, H, W)
    return out, {
        "angle_deg": round(angle, 3),
        "degrees_range": "(-30, 30)",
        "fill": 192,
        "interpolation": "bilinear",
    }


def tf_crop(clip, rng, seed):
    crop = (0.1, 0.1, 0.2, 0.1)
    _, _, H, W = clip.shape
    x_left = int(W * rng.uniform(0, crop[0]))
    x_right = W - int(W * rng.uniform(0, crop[1]))
    y_bottom = int(H * rng.uniform(0, crop[2]))
    y_top = H - int(H * rng.uniform(0, crop[3]))
    y_bottom, y_top = min(y_bottom, y_top), max(y_bottom, y_top)
    x_left, x_right = min(x_left, x_right), max(x_left, x_right)
    out = clip[..., y_bottom:y_top, x_left:x_right].contiguous()
    return out, {
        "crop_ratios_LRBT": crop,
        "x_left": x_left,
        "x_right": x_right,
        "y_bottom": y_bottom,
        "y_top": y_top,
        "out_HW": (out.shape[2], out.shape[3]),
    }


def tf_resize(clip, rng, seed):
    out = F.interpolate(
        clip.unsqueeze(0), (8, 256, 256), mode="trilinear", align_corners=True
    )[0]
    return out, {
        "t": 8,
        "visual": (256, 256),
        "note": "no-op here since input is already 8x256x256",
    }


def tf_znorm(clip, rng, seed):
    mean, std = float(clip.mean()), float(clip.std())
    chwd = clip.permute(0, 2, 3, 1).contiguous()
    subject = tio.Subject(image=tio.ScalarImage(tensor=chwd))
    out = tio.ZNormalization()(subject)
    znorm = out.image.data.permute(0, 3, 1, 2).contiguous() / 255.0
    disp = znorm - znorm.min()
    disp = disp / (disp.max() + 1e-8) * 255.0  # min-max rescale for viewing only
    return disp, {
        "input_mean": round(mean, 3),
        "input_std": round(std, 3),
        "operation": "(x - mean) / std, then / 255",
        "div255": True,
        "note": "frames min-max rescaled for display; actual output is z-scored",
    }


# Order matches Data/dataset.py::train_trans.
TRANSFORMS = [
    ("01_RandomErode", tf_erode),
    ("02_RandomDilate", tf_dilate),
    ("03_TioClamp", tf_clamp),
    ("04_TioRandomFlip", tf_flip),
    ("05_TioRandomAnisotropy", tf_anisotropy),
    ("06_TioRandomMotion", tf_motion),
    ("07_TioRandomGhosting", tf_ghosting),
    ("08_TioRandomSpike", tf_spike),
    ("09_TioRandomBiasField", tf_biasfield),
    ("10_TioRandomBlur", tf_blur),
    ("11_TioRandomNoise", tf_noise),
    ("12_TioRandomGamma", tf_gamma),
    ("13_RandomRotation", tf_rotation),
    ("14_Crop", tf_crop),
    ("15_Resize", tf_resize),
    ("16_TioZNormalization", tf_znorm),
]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def iter_subjects(limit=None):
    """Enumerate (label, subject_id) pairs from the CVFS_paper index folder."""
    for label in LABELS:
        label_dir = os.path.join(INDEX_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        sids = sorted(d for d in os.listdir(label_dir)
                      if os.path.isdir(os.path.join(label_dir, d)))
        if limit is not None:
            sids = sids[:limit]
        for sid in sids:
            yield label, sid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max subjects per label (default: all).",
    )
    args = parser.parse_args()

    import random

    subjects = list(iter_subjects(args.limit))
    print("Applying %d transforms to %d subjects x %d views (from NIfTI) -> %s"
          % (len(TRANSFORMS), len(subjects), len(VIEWS),
             os.path.relpath(OUT_DIR, ROOT)))

    for label, sid in subjects:
        for view in VIEWS:
            clip, meta = load_clip(sid, view)
            for tname, tfunc in TRANSFORMS:
                seed = leaf_seed(tname, label, sid, view)
                rng = random.Random(seed)
                out_clip, params = tfunc(clip.clone(), rng, seed)
                params = {"seed": seed, **meta, **params}
                out_dir = os.path.join(OUT_DIR, tname, label, sid, view)
                save_clip(out_clip, out_dir, params, tname)
        print("  done %-4s %s" % (label, sid))

    print("Finished.")


if __name__ == "__main__":
    main()
