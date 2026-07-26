"""Plot a stacked bar chart of scan counts by frame size for DSA_public_nifti.

Modeled on ``plot_frame_size_counts.py`` (AmTICIS size × TICI stacking) and
``generate_dias_frame_size_barplot.py`` (size × split stacking). Reads NIfTI
volumes under ``DSA_public_nifti`` and plots how many scans have a given spatial
frame size (H x W), colored by train (``imagesTr``) vs test (``imagesTs``).

Usage:
    uv run python plotting_functions/generate_dsa_public_frame_size_barplot.py
    uv run python plotting_functions/generate_dsa_public_frame_size_barplot.py \\
        --data-dir DSA_public_nifti --output plots/dsa_public_frame_size_barplot.png
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib

# Same palette slots as the DIAS / DSA_public frame-count plots (blue / magenta).
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
        default=Path("plots/dsa_public_frame_size_barplot.png"),
        help="Output plot file path (default: plots/dsa_public_frame_size_barplot.png).",
    )
    return parser.parse_args()


def detect_split(nii_path: Path, data_dir: Path) -> str:
    """Return the dataset split folder: the first path component under data_dir."""
    return nii_path.relative_to(data_dir).parts[0]


def get_spatial_size(nii_path: Path) -> tuple[int, int]:
    """Return spatial frame size (H, W) without loading pixel data."""
    shape = nib.load(nii_path).shape
    return int(shape[0]), int(shape[1])


def size_key(size_str: str) -> tuple[int, int]:
    h_str, w_str = size_str.split("x")
    return int(h_str), int(w_str)


def collect_counts(data_dir: Path) -> tuple[dict[str, Counter], list[str]]:
    """Return {split: Counter({"HxW": num_scans})} and ordered split names."""
    nii_files = sorted(data_dir.rglob("*.nii*"))
    if not nii_files:
        raise ValueError(f"No NIfTI files found under: {data_dir}")

    counts: dict[str, Counter] = defaultdict(Counter)
    for nii_path in nii_files:
        split = detect_split(nii_path, data_dir)
        h, w = get_spatial_size(nii_path)
        counts[split][f"{h}x{w}"] += 1

    found_splits = set(counts)
    unexpected = sorted(found_splits - set(SPLIT_COLORS))
    if unexpected:
        raise ValueError(
            f"No color assigned for split(s) {unexpected}; add them to SPLIT_COLORS."
        )
    splits = [s for s in SPLIT_ORDER if s in found_splits]
    return counts, splits


def plot_counts(counts: dict[str, Counter], splits: list[str], output_path: Path) -> None:
    all_sizes = sorted(
        {size for split_counter in counts.values() for size in split_counter},
        key=size_key,
    )

    plt.figure(figsize=(12, 6))
    bottoms = [0] * len(all_sizes)
    for split in splits:
        values = [counts[split].get(size, 0) for size in all_sizes]
        plt.bar(
            all_sizes,
            values,
            bottom=bottoms,
            color=SPLIT_COLORS[split],
            edgecolor="black",
            linewidth=0.5,
            label=SPLIT_LABELS[split],
        )
        bottoms = [b + v for b, v in zip(bottoms, values)]

    plt.title("Scan Count by Frame Size Across DSA_public Scans")
    plt.xlabel("Frame size (H x W)")
    plt.ylabel("Number of scans")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(title="Split")

    for i, total in enumerate(bottoms):
        if total > 0:
            plt.text(i, total + 0.5, str(total), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    counts, splits = collect_counts(args.data_dir)
    plot_counts(counts, splits, args.output)

    total_scans = sum(sum(c.values()) for c in counts.values())
    unique_sizes = len({size for c in counts.values() for size in c})
    print(f"Saved plot to: {args.output}")
    print(f"Total scans processed: {total_scans}")
    for split in splits:
        print(f"  {SPLIT_LABELS[split]} ({split}): {sum(counts[split].values())}")
    print(f"Unique frame sizes: {unique_sizes}")


if __name__ == "__main__":
    main()
