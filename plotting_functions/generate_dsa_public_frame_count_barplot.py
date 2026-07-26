"""Generate a stacked bar plot of frame-count frequency for DSA_public_nifti.

Modeled on ``generate_frame_count_barplot.py`` (AmTICIS view stacking) and
``generate_dias_frame_count_barplot.py`` (split stacking). Reads NIfTI volumes
under ``DSA_public_nifti`` and plots how many scans have a given frame count,
colored by train (``imagesTr``) vs test (``imagesTs``).

Usage:
    uv run python plotting_functions/generate_dsa_public_frame_count_barplot.py
    uv run python plotting_functions/generate_dsa_public_frame_count_barplot.py \\
        --data-dir DSA_public_nifti --output plots/dsa_public_frame_count_barplot.png
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib

# Same palette slots as the DIAS split plots (blue / magenta).
SPLIT_COLORS = {
    "imagesTr": "#2a78d6",
    "imagesTs": "#e87ba4",
}
SPLIT_LABELS = {
    "imagesTr": "Train",
    "imagesTs": "Test",
}
SPLIT_ORDER = ("imagesTr", "imagesTs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("DSA_public_nifti"),
        help="Directory of converted DSA_public NIfTI scans (default: DSA_public_nifti).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/dsa_public_frame_count_barplot.png"),
        help="Output plot file path (default: plots/dsa_public_frame_count_barplot.png).",
    )
    return parser.parse_args()


def detect_split(nii_path: Path, data_dir: Path) -> str:
    """Return the dataset split folder: the first path component under data_dir."""
    return nii_path.relative_to(data_dir).parts[0]


def get_frame_count(nii_path: Path) -> int:
    """Return the number of frames (3rd axis) without loading pixel data."""
    return int(nib.load(nii_path).shape[2])


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    nii_files = sorted(args.data_dir.rglob("*.nii*"))
    if not nii_files:
        raise ValueError(f"No NIfTI files found under: {args.data_dir}")

    found_splits = {detect_split(p, args.data_dir) for p in nii_files}
    unexpected = sorted(found_splits - set(SPLIT_COLORS))
    if unexpected:
        raise ValueError(
            f"No color assigned for split(s) {unexpected}; add them to SPLIT_COLORS."
        )

    splits = [s for s in SPLIT_ORDER if s in found_splits]
    frame_counts: dict[str, Counter[int]] = {split: Counter() for split in splits}
    for nii_path in nii_files:
        split = detect_split(nii_path, args.data_dir)
        frame_counts[split][get_frame_count(nii_path)] += 1

    x_vals = sorted(set().union(*(frame_counts[s].keys() for s in splits)))
    per_split_vals = {split: [frame_counts[split][x] for x in x_vals] for split in splits}

    plt.figure(figsize=(14, 6))
    bottoms = [0] * len(x_vals)
    for split in splits:
        vals = per_split_vals[split]
        plt.bar(
            x_vals,
            vals,
            bottom=bottoms,
            color=SPLIT_COLORS[split],
            edgecolor="black",
            linewidth=0.5,
            label=SPLIT_LABELS[split],
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    plt.title("Frame Count Distribution Across DSA_public Scans")
    plt.xlabel("Number of Frames per Scan")
    plt.ylabel("Number of Scans")
    plt.xticks(x_vals, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(title="Split")

    for i, total in enumerate(bottoms):
        if total > 0:
            plt.text(x_vals[i], total + 0.5, str(total), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    plt.close()

    print(f"Saved bar plot to: {args.output}")
    print(f"Total scans processed: {len(nii_files)}")
    for split in splits:
        print(f"  {SPLIT_LABELS[split]} ({split}): {sum(per_split_vals[split])}")
    print(f"Unique frame counts: {len(x_vals)}")


if __name__ == "__main__":
    main()
