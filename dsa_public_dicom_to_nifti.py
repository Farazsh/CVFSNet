#!/usr/bin/env python3
"""Convert DICOM files under DSA_public to gzip-compressed NIfTI (``.nii.gz``).

Recursively finds ``.dcm`` / ``.dicom`` files under the input directory and
writes one NIfTI volume per DICOM file, mirroring the relative folder layout
into the output directory.

No preprocessing is applied: pixel values, spacing, origin, and direction are
passed through as-is via SimpleITK read/write. There is no resampling,
intensity scaling, cropping, or dtype casting beyond what the NIfTI writer
needs for a lossless round-trip of the array already loaded from DICOM.

Usage
-----
    # Convert everything under ./DSA_public into ./DSA_public_nifti
    uv run python dsa_public_dicom_to_nifti.py

    # Custom paths / dry-run
    uv run python dsa_public_dicom_to_nifti.py \\
        --input-dir DSA_public --output-dir DSA_public_nifti --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import SimpleITK as sitk

LOGGER = logging.getLogger("dsa_public_dicom_to_nifti")

DEFAULT_EXTENSIONS = (".dcm", ".dicom")


@dataclass
class ConversionResult:
    source: Path
    output_path: Optional[Path]
    status: str  # "converted" | "skipped_existing" | "error"
    message: str = ""
    size: tuple[int, ...] = ()


def discover_dicoms(input_dir: Path, extensions: Iterable[str]) -> list[Path]:
    """Recursively collect DICOM files under input_dir."""
    extensions = {ext.lower() for ext in extensions}
    paths = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(paths)


def output_path_for(source: Path, input_dir: Path, output_dir: Path) -> Path:
    relative = source.relative_to(input_dir)
    return output_dir / relative.with_suffix(".nii.gz")


def convert_one(
    source: Path,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> ConversionResult:
    out_path = output_path_for(source, input_dir, output_dir)

    if out_path.exists() and not overwrite:
        return ConversionResult(source, out_path, "skipped_existing")

    try:
        image = sitk.ReadImage(str(source))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # WriteImage with a .nii.gz destination stores the loaded array and
        # geometry metadata without resampling or intensity transforms.
        sitk.WriteImage(image, str(out_path), useCompression=True)
    except Exception as exc:  # noqa: BLE001 - report and continue
        return ConversionResult(source, None, "error", str(exc))

    return ConversionResult(
        source,
        out_path,
        "converted",
        size=tuple(image.GetSize()),
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("DSA_public"),
        help="Root directory to scan recursively for DICOM files (default: DSA_public).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("DSA_public_nifti"),
        help="Directory to write .nii.gz files into, mirroring the input layout "
             "(default: DSA_public_nifti).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help=f"File extensions to treat as DICOM (default: {' '.join(DEFAULT_EXTENSIONS)}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (default: 1). Increase for faster conversion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite NIfTI files that already exist (default: skip them).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List DICOM -> NIfTI mappings without reading or writing files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
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

    extensions = [
        ext if ext.startswith(".") else f".{ext}" for ext in args.extensions
    ]

    LOGGER.info("Scanning %s for DICOM files...", args.input_dir)
    dicoms = discover_dicoms(args.input_dir, extensions)
    LOGGER.info("Found %d DICOM file(s).", len(dicoms))

    if not dicoms:
        LOGGER.warning("Nothing to convert.")
        return 0

    if args.dry_run:
        for source in dicoms:
            out_path = output_path_for(source, args.input_dir, args.output_dir)
            LOGGER.info("[dry-run] %s -> %s", source, out_path)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ConversionResult] = []
    if args.workers <= 1:
        for source in dicoms:
            result = convert_one(
                source, args.input_dir, args.output_dir, args.overwrite
            )
            if result.status == "converted":
                LOGGER.info(
                    "Converted %s -> %s (size=%s)",
                    result.source,
                    result.output_path,
                    result.size,
                )
            elif result.status == "skipped_existing":
                LOGGER.debug("Skipped existing %s", result.output_path)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    convert_one,
                    source,
                    args.input_dir,
                    args.output_dir,
                    args.overwrite,
                ): source
                for source in dicoms
            }
            for future in as_completed(futures):
                result = future.result()
                if result.status == "converted":
                    LOGGER.info(
                        "Converted %s -> %s (size=%s)",
                        result.source,
                        result.output_path,
                        result.size,
                    )
                results.append(result)

    converted = [r for r in results if r.status == "converted"]
    skipped = [r for r in results if r.status == "skipped_existing"]
    errors = [r for r in results if r.status == "error"]

    for result in errors:
        LOGGER.error("Failed to convert %s: %s", result.source, result.message)

    LOGGER.info(
        "Done. converted=%d skipped_existing=%d errors=%d",
        len(converted),
        len(skipped),
        len(errors),
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
