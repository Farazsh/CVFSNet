"""Generate a stacked bar plot of frame-count frequency across AmTICIS scans.

Each bar shows how many scans have a given frame count, split by view:
  - Coronal (_C) in blue
  - Sagittal (_S) in green

Usage:
    uv run python generate_frame_count_barplot.py
    uv run python generate_frame_count_barplot.py --data-dir AmTICIS --output frame_count_barplot.png
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import SimpleITK as sitk

CORONAL_COLOR = "#3A86FF"
SAGITTAL_COLOR = "#59A14F"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("AmTICIS"),
        help="Directory containing .nii.gz scans (default: AmTICIS).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("frame_count_barplot.png"),
        help="Output plot file path (default: frame_count_barplot.png).",
    )
    return parser.parse_args()


def detect_view(filename: str) -> str | None:
    """Return 'coronal' or 'sagittal' based on the _C/_S marker in the filename."""
    match = re.search(r"_(C|S)(?:#|_)", filename)
    if not match:
        return None
    return "coronal" if match.group(1) == "C" else "sagittal"


def get_frame_count(nii_path: Path) -> int:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(nii_path)))  # (H, W, T)
    return int(arr.shape[2])


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    nii_files = sorted(args.data_dir.glob("*.nii.gz"))
    if not nii_files:
        raise ValueError(f"No .nii.gz files found in: {args.data_dir}")

    frame_counts: dict[str, Counter[int]] = {
        "coronal": Counter(),
        "sagittal": Counter(),
    }
    skipped = 0
    for nii in nii_files:
        view = detect_view(nii.name)
        if view is None:
            skipped += 1
            continue
        frame_counts[view][get_frame_count(nii)] += 1

    if not frame_counts["coronal"] and not frame_counts["sagittal"]:
        raise ValueError(
            f"No scans with _C or _S view markers found in: {args.data_dir}"
        )

    x_vals = sorted(set(frame_counts["coronal"]) | set(frame_counts["sagittal"]))
    coronal_vals = [frame_counts["coronal"][x] for x in x_vals]
    sagittal_vals = [frame_counts["sagittal"][x] for x in x_vals]

    plt.figure(figsize=(14, 6))
    plt.bar(
        x_vals,
        coronal_vals,
        color=CORONAL_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label="Coronal",
    )
    plt.bar(
        x_vals,
        sagittal_vals,
        bottom=coronal_vals,
        color=SAGITTAL_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label="Sagittal",
    )
    plt.title("Frame Count Distribution Across AmTICIS Scans")
    plt.xlabel("Number of Frames per Scan")
    plt.ylabel("Number of Scans")
    plt.xticks(x_vals, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(title="View")

    for i, x in enumerate(x_vals):
        total = coronal_vals[i] + sagittal_vals[i]
        if total > 0:
            plt.text(
                x,
                total + 0.5,
                str(total),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    plt.close()

    processed = sum(coronal_vals) + sum(sagittal_vals)
    print(f"Saved bar plot to: {args.output}")
    print(f"Total scans processed: {processed}")
    print(f"  Coronal: {sum(coronal_vals)}")
    print(f"  Sagittal: {sum(sagittal_vals)}")
    if skipped:
        print(f"Skipped scans (no _C/_S marker): {skipped}")
    print(f"Unique frame counts: {len(x_vals)}")


if __name__ == "__main__":
    main()
