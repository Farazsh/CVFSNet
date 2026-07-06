"""Generate a bar plot of frame-count frequency across AmTICIS scans.

Usage:
    uv run python generate_frame_count_barplot.py
    uv run python generate_frame_count_barplot.py --data-dir AmTICIS --output frame_count_barplot.png
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import SimpleITK as sitk


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

    frame_counter: Counter[int] = Counter()
    for nii in nii_files:
        frame_counter[get_frame_count(nii)] += 1

    x_vals = sorted(frame_counter.keys())
    y_vals = [frame_counter[x] for x in x_vals]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(x_vals, y_vals, color="#3A86FF", edgecolor="black", linewidth=0.5)
    plt.title("Frame Count Distribution Across AmTICIS Scans")
    plt.xlabel("Number of Frames per Scan")
    plt.ylabel("Number of Scans")
    plt.xticks(x_vals, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    # Annotate bars with scan counts for readability.
    for rect, y in zip(bars, y_vals):
        plt.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.5,
            str(y),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    plt.close()

    print(f"Saved bar plot to: {args.output}")
    print(f"Total scans processed: {len(nii_files)}")
    print(f"Unique frame counts: {len(x_vals)}")


if __name__ == "__main__":
    main()
