"""Plot stacked frame-size scan counts for AP and sagittal views.

This script scans NIfTI files in the AmTICIS folder, extracts each scan's
spatial frame size (H x W) and TICI label, and builds two stacked bar plots:
  1) AP (coronal, `_C`)
  2) sagittal (`_S`)
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="AmTICIS",
        help="Directory containing .nii.gz scan files (default: AmTICIS).",
    )
    parser.add_argument(
        "--output",
        default="frame_size_barplot.png",
        help="Output image path for the barplot (default: frame_size_barplot.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window in addition to saving the image.",
    )
    return parser.parse_args()


def detect_view(filename: str) -> str | None:
    """Map filename to AP/sagittal based on _C/_S marker."""
    match = re.search(r"_(C|S)(?:#|_)", filename)
    if not match:
        return None
    return "AP" if match.group(1) == "C" else "sagittal"


def get_spatial_size(path: str) -> Tuple[int, int]:
    """Return spatial frame size (H, W) from a NIfTI file."""
    # In this project, arrays read as (H, W, T).
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
    return int(arr.shape[0]), int(arr.shape[1])


def extract_label(filename: str) -> str | None:
    """Extract TICI label (e.g., T0, T1, T2A, T2B, T3) from filename."""
    parts = filename.split("_")
    if len(parts) < 2:
        return None
    label = parts[1].upper()
    if label in {"T0", "T1", "T2A", "T2B", "T3"}:
        return label
    return None


def collect_counts(data_dir: str) -> Tuple[Dict[str, Dict[str, Counter]], List[str]]:
    # counts[view][size][label] = number of scans
    counts = {
        "AP": defaultdict(Counter),
        "sagittal": defaultdict(Counter),
    }
    labels_seen = set()
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".nii.gz"):
            continue
        view = detect_view(name)
        if view is None:
            continue
        label = extract_label(name)
        if label is None:
            continue
        labels_seen.add(label)
        h, w = get_spatial_size(os.path.join(data_dir, name))
        counts[view][f"{h}x{w}"][label] += 1
    labels = sorted(labels_seen, key=lambda x: ["T0", "T1", "T2A", "T2B", "T3"].index(x))
    return counts, labels


def size_key(size_str: str) -> Tuple[int, int]:
    h_str, w_str = size_str.split("x")
    return int(h_str), int(w_str)


def plot_counts(
    counts: Dict[str, Dict[str, Counter]], labels: List[str], output_path: str, show: bool
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    views = ["AP", "sagittal"]
    label_colors = {
        "T0": "#4E79A7",
        "T1": "#F28E2B",
        "T2A": "#E15759",
        "T2B": "#76B7B2",
        "T3": "#59A14F",
    }

    for ax, view in zip(axes, views):
        size_to_label_counter = counts[view]
        if not size_to_label_counter:
            ax.set_title(f"{view} view (no scans found)")
            ax.axis("off")
            continue

        sizes = sorted(size_to_label_counter.keys(), key=size_key)
        bottoms = [0] * len(sizes)

        for label in labels:
            values = [size_to_label_counter[s].get(label, 0) for s in sizes]
            ax.bar(
                sizes,
                values,
                bottom=bottoms,
                color=label_colors.get(label, "#999999"),
                edgecolor="black",
                linewidth=0.5,
                label=label,
            )
            bottoms = [b + v for b, v in zip(bottoms, values)]

        ax.set_title(f"{view} view")
        ax.set_xlabel("Frame size (H x W)")
        ax.set_ylabel("Number of scans")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

        for i, v in enumerate(bottoms):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    # Single legend shared across both subplots with consistent label colors.
    handles, legend_labels = axes[0].get_legend_handles_labels()
    dedup = {}
    for h, l in zip(handles, legend_labels):
        dedup[l] = h
    fig.legend(
        dedup.values(),
        dedup.keys(),
        title="TICI label",
        loc="upper left",
        ncol=1,
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )

    fig.suptitle("Scan count by frame size (stacked by TICI label)", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    counts, labels = collect_counts(args.data_dir)
    plot_counts(counts, labels, args.output, args.show)
    print(f"Saved plot to: {args.output}")
    print(f"TICI labels found: {', '.join(labels)}")
    print(f"AP unique sizes: {len(counts['AP'])}")
    print(f"sagittal unique sizes: {len(counts['sagittal'])}")


if __name__ == "__main__":
    main()
