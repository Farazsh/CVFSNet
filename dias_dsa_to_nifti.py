#!/usr/bin/env python3
"""Convert per-frame DSA (Digital Subtraction Angiography) images into NIfTI volumes.

The DIAS dataset stores each DSA acquisition as a sequence of individual frame
images spread across several dataset splits (e.g. ``training``, ``validation``,
``test``, ``unlabeled_DSA``). Filenames encode two numbers: which scan the frame
belongs to, and the frame's position within that scan's acquisition sequence
(e.g. ``image_s12_i4.png`` -> scan 12, frame 4). The exact naming convention is
not guaranteed to be identical across every split, so this script infers the
(scan, frame) pair from the digits present in each filename rather than
hard-coding one regular expression.

For every directory it walks, the script:
  1. Groups files in that directory by scan id.
  2. Orders each group's frames numerically (not lexicographically) by the
     frame index embedded in the filename.
  3. Validates that all frames in a group share the same shape/dtype and that
     frame indices are unique, refusing to guess when data looks inconsistent.
  4. Stacks the frames into a single ``(H, W, num_frames)`` volume and writes
     it out as a NIfTI file, mirroring the input directory layout under the
     output directory.

Design choices made to avoid losing information:
  - Pixel data is read with its native bit depth (e.g. uint8) and never
    rescaled, normalized, resized, or reinterpolated.
  - Frames are only collapsed from RGB(A) to single-channel when every color
    channel (and the alpha channel, if present) is byte-for-byte identical --
    i.e. when the "color" image is provably a grayscale image stored
    redundantly. If channels genuinely differ, the frame is rejected rather
    than lossily averaged/guessed.
  - Output volumes are written as gzip-compressed NIfTI (``.nii.gz``), which
    is lossless (unlike, say, re-encoding to JPEG).
  - Because individual PNG frames carry no real-world pixel-spacing/DICOM
    geometry, the NIfTI affine is set to identity and the qform/sform codes
    are explicitly marked "unknown" (0) rather than fabricating spatial
    calibration that was never present in the source data.

Usage
-----
    # Convert everything under ./DIAS into ./DIAS_nifti
    python dias_dsa_to_nifti.py --input-dir DIAS --output-dir DIAS_nifti

    # Preview what would be grouped/converted without writing any files
    python dias_dsa_to_nifti.py --input-dir DIAS --output-dir DIAS_nifti --dry-run

    # Use a custom filename pattern (must define named groups "scan"/"frame")
    python dias_dsa_to_nifti.py --input-dir DIAS --output-dir DIAS_nifti \\
        --pattern "image_s(?P<scan>\\d+)_i(?P<frame>\\d+)"

The script has no dependency on the specific DIAS folder names -- point
``--input-dir`` at any directory tree containing per-frame scan images and it
will recurse into it.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import nibabel as nib
import numpy as np
from PIL import Image

LOGGER = logging.getLogger("dias_dsa_to_nifti")

DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Fallback filename parser used when --pattern is not supplied: treat the
# FIRST run of digits in the filename as the scan id and the LAST run of
# digits as the frame index. This matches "image_s<scan>_i<frame>.png" style
# names generically, without assuming a fixed separator/prefix, so it keeps
# working even if a different split uses a different naming scheme.
_DIGIT_RUN_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class Frame:
    """A single image file and the frame index parsed from its name."""

    index: int
    path: Path


@dataclass
class ScanGroup:
    """All frames belonging to one scan, found in one source directory."""

    directory: Path
    scan_id: str
    frames: list[Frame] = field(default_factory=list)


@dataclass
class ConversionResult:
    group_key: str
    output_path: Optional[Path]
    status: str  # "converted" | "skipped_existing" | "skipped_invalid" | "error"
    num_frames: int = 0
    message: str = ""


# --------------------------------------------------------------------------
# Discovery: walk the input tree and group frame files by (directory, scan id)
# --------------------------------------------------------------------------


def parse_scan_and_frame(
    filename_stem: str, pattern: Optional[re.Pattern]
) -> Optional[tuple[str, int]]:
    """Extract (scan_id, frame_index) from a filename stem, or None if it can't be parsed."""
    if pattern is not None:
        match = pattern.search(filename_stem)
        if not match:
            return None
        try:
            return match.group("scan"), int(match.group("frame"))
        except (IndexError, ValueError):
            LOGGER.warning(
                "Pattern matched %r but 'scan'/'frame' groups were not both "
                "present/numeric; skipping.",
                filename_stem,
            )
            return None

    digit_runs = _DIGIT_RUN_RE.findall(filename_stem)
    if len(digit_runs) < 2:
        # A single number (or none) means this filename doesn't encode a
        # (scan, frame) pair -- e.g. a per-scan label mask like "label_s2.png".
        return None
    return digit_runs[0], int(digit_runs[-1])


def discover_scan_groups(
    input_dir: Path,
    extensions: Iterable[str],
    pattern: Optional[re.Pattern],
) -> list[ScanGroup]:
    """Recursively find image files under input_dir and group them by (dir, scan id)."""
    extensions = {ext.lower() for ext in extensions}
    groups: dict[tuple[Path, str], ScanGroup] = {}
    num_skipped = 0

    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        parsed = parse_scan_and_frame(path.stem, pattern)
        if parsed is None:
            num_skipped += 1
            continue
        scan_id, frame_index = parsed
        key = (path.parent, scan_id)
        group = groups.setdefault(
            key, ScanGroup(directory=path.parent, scan_id=scan_id)
        )
        group.frames.append(Frame(index=frame_index, path=path))

    if num_skipped:
        LOGGER.info(
            "Skipped %d file(s) whose name did not encode a (scan, frame) pair "
            "(e.g. single-file label masks) -- run with -v to see each one.",
            num_skipped,
        )
    return list(groups.values())


def validate_and_sort_group(group: ScanGroup) -> Optional[ScanGroup]:
    """Order frames by numeric index and reject groups with ambiguous ordering.

    Returns None (and logs why) if the group cannot be safely converted, e.g.
    because two files claim the same frame index -- silently picking one
    would discard data without the user knowing.
    """
    indices = [frame.index for frame in group.frames]
    if len(indices) != len(set(indices)):
        LOGGER.error(
            "Scan '%s' in %s has duplicate frame indices %s; refusing to "
            "guess an order -- skipping this scan.",
            group.scan_id,
            group.directory,
            sorted(idx for idx in indices if indices.count(idx) > 1),
        )
        return None

    group.frames.sort(key=lambda frame: frame.index)

    expected = set(range(min(indices), max(indices) + 1))
    missing = sorted(expected - set(indices))
    if missing:
        LOGGER.warning(
            "Scan '%s' in %s is missing frame index(es) %s; the %d frame(s) "
            "present will still be stacked in order.",
            group.scan_id,
            group.directory,
            missing,
            len(indices),
        )
    return group


# --------------------------------------------------------------------------
# Pixel loading: preserve native bit depth, never guess a lossy conversion
# --------------------------------------------------------------------------


def load_frame_array(path: Path) -> np.ndarray:
    """Load one frame as a 2-D array, preserving its native bit depth exactly.

    Multi-channel PNGs (RGB/RGBA) are only collapsed to a single channel when
    every channel (and alpha, if present) is provably redundant -- i.e. the
    image is grayscale data stored with duplicated channels. If channels
    genuinely disagree, this raises rather than silently averaging/guessing,
    since that would discard real information.
    """
    with Image.open(path) as img:
        if img.mode == "P":
            # Palette images must be resolved to real pixel values before we
            # can inspect channels; this is a required decode step, not a
            # lossy approximation.
            img = img.convert("RGB")
        arr = np.asarray(img)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        color_channels = [arr[..., c] for c in range(min(arr.shape[2], 3))]
        if not all(np.array_equal(color_channels[0], c) for c in color_channels[1:]):
            raise ValueError(
                f"{path}: R/G/B channels differ -- this is not a redundantly "
                "encoded grayscale image, refusing to lossily flatten it"
            )
        if arr.shape[2] == 4:
            alpha = arr[..., 3]
            if alpha.min() != alpha.max():
                raise ValueError(
                    f"{path}: alpha channel is not constant -- it may carry "
                    "real information, refusing to discard it"
                )
        return color_channels[0]

    raise ValueError(f"{path}: unsupported array shape {arr.shape}")


def build_volume(group: ScanGroup) -> np.ndarray:
    """Stack a scan's already-ordered frames into a (H, W, num_frames) volume."""
    arrays: list[np.ndarray] = []
    ref_shape: Optional[tuple[int, ...]] = None
    ref_dtype: Optional[np.dtype] = None

    for frame in group.frames:
        arr = load_frame_array(frame.path)
        if ref_shape is None:
            ref_shape, ref_dtype = arr.shape, arr.dtype
        elif arr.shape != ref_shape:
            raise ValueError(
                f"{frame.path}: shape {arr.shape} != {ref_shape} of the "
                f"first frame in scan '{group.scan_id}' -- refusing to resize"
            )
        elif arr.dtype != ref_dtype:
            raise ValueError(
                f"{frame.path}: dtype {arr.dtype} != {ref_dtype} of the "
                f"first frame in scan '{group.scan_id}'"
            )
        arrays.append(arr)

    return np.stack(arrays, axis=-1)


# --------------------------------------------------------------------------
# NIfTI packaging
# --------------------------------------------------------------------------


def volume_to_nifti(volume: np.ndarray, group: ScanGroup) -> nib.Nifti1Image:
    """Wrap a stacked volume in a NIfTI image without fabricating spatial metadata.

    Individual DSA frame PNGs carry no real-world pixel-spacing/orientation
    information, so an identity affine is used (voxel indices map straight
    to output coordinates, 1 unit apart) rather than inventing calibrated mm
    spacing that was never present in the source data. The qform is
    explicitly marked "unknown" (code 0). Note that nibabel always persists
    the sform as "aligned" (code 2) whenever an affine is supplied -- that's
    expected/unavoidable round-trip behavior, not a claim of real-world
    calibration; it still refers to the same identity affine. The third axis
    is the acquisition frame index, not a physical slice location -- this is
    recorded in the header's free-text description field.
    """
    affine = np.eye(4, dtype=np.float64)
    img = nib.Nifti1Image(volume, affine)
    img.header.set_zooms((1.0, 1.0, 1.0))
    img.header.set_qform(affine, code=0)
    descrip = f"DIAS DSA scan {group.scan_id}; axis2=frame index, not spatial Z"
    img.header["descrip"] = descrip.encode("ascii", "ignore")[:80]
    return img


# --------------------------------------------------------------------------
# Per-group conversion (safe to run in a worker process)
# --------------------------------------------------------------------------


def output_path_for_group(group: ScanGroup, input_dir: Path, output_dir: Path, compress: bool) -> Path:
    relative_dir = group.directory.relative_to(input_dir)
    suffix = ".nii.gz" if compress else ".nii"
    return output_dir / relative_dir / f"scan_{group.scan_id}{suffix}"


def convert_group(
    group: ScanGroup,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
    compress: bool,
) -> ConversionResult:
    key = f"{group.directory}/scan_{group.scan_id}"
    out_path = output_path_for_group(group, input_dir, output_dir, compress)

    if out_path.exists() and not overwrite:
        return ConversionResult(key, out_path, "skipped_existing", len(group.frames))

    validated = validate_and_sort_group(group)
    if validated is None:
        return ConversionResult(key, None, "skipped_invalid", len(group.frames))

    try:
        volume = build_volume(validated)
        nifti_img = volume_to_nifti(volume, validated)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nifti_img.to_filename(str(out_path))
    except Exception as exc:  # noqa: BLE001 - report and continue with other scans
        return ConversionResult(key, None, "error", len(group.frames), str(exc))

    return ConversionResult(key, out_path, "converted", len(group.frames))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("DIAS"),
        help="Root directory to scan recursively for per-frame scan images (default: DIAS).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("DIAS_nifti"),
        help="Directory to write NIfTI volumes into, mirroring the input layout (default: DIAS_nifti).",
    )
    parser.add_argument(
        "--extensions", nargs="+", default=list(DEFAULT_EXTENSIONS),
        help=f"File extensions to treat as frame images (default: {' '.join(DEFAULT_EXTENSIONS)}).",
    )
    parser.add_argument(
        "--pattern", type=str, default=None,
        help="Optional regex with named groups 'scan' and 'frame' to override "
             "the default digit-based filename parser, e.g. "
             r"'image_s(?P<scan>\d+)_i(?P<frame>\d+)'.",
    )
    parser.add_argument(
        "--min-frames", type=int, default=1,
        help="Skip scans with fewer than this many frames (default: 1, i.e. keep everything).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel worker processes (default: all CPUs). Use 1 to run serially, "
             "which gives clearer tracebacks when debugging.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute and overwrite NIfTI files that already exist (default: skip them).",
    )
    parser.add_argument(
        "--no-compress", action="store_true",
        help="Write uncompressed .nii instead of gzip-compressed .nii.gz.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover and validate scan groups, then report what would be written, "
             "without reading pixel data or writing any files.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input_dir.is_dir():
        LOGGER.error("Input directory does not exist: %s", args.input_dir)
        return 1

    pattern = re.compile(args.pattern) if args.pattern else None
    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in args.extensions]

    LOGGER.info("Scanning %s for per-frame scan images...", args.input_dir)
    groups = discover_scan_groups(args.input_dir, extensions, pattern)
    groups = [g for g in groups if len(g.frames) >= args.min_frames]
    LOGGER.info("Found %d scan(s) across %d frame file(s).", len(groups), sum(len(g.frames) for g in groups))

    if args.dry_run:
        for group in sorted(groups, key=lambda g: (str(g.directory), g.scan_id)):
            out_path = output_path_for_group(group, args.input_dir, args.output_dir, not args.no_compress)
            LOGGER.info(
                "[dry-run] %s -> %s (%d frames)", group.directory / f"scan_{group.scan_id}",
                out_path, len(group.frames),
            )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ConversionResult] = []
    if args.workers == 1:
        for group in groups:
            results.append(
                convert_group(group, args.input_dir, args.output_dir, args.overwrite, not args.no_compress)
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    convert_group, group, args.input_dir, args.output_dir, args.overwrite, not args.no_compress
                ): group
                for group in groups
            }
            for future in as_completed(futures):
                results.append(future.result())

    converted = [r for r in results if r.status == "converted"]
    skipped_existing = [r for r in results if r.status == "skipped_existing"]
    skipped_invalid = [r for r in results if r.status == "skipped_invalid"]
    errors = [r for r in results if r.status == "error"]

    for result in errors:
        LOGGER.error("Failed to convert %s: %s", result.group_key, result.message)

    LOGGER.info(
        "Done. converted=%d skipped_existing=%d skipped_invalid=%d errors=%d "
        "(total frames written: %d)",
        len(converted), len(skipped_existing), len(skipped_invalid), len(errors),
        sum(r.num_frames for r in converted),
    )

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
