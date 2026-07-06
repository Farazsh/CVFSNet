"""Generate a flowchart/tree diagram summarizing AmTICIS scans.

The diagram includes:
1) Number of scans per view (AP vs sagittal)
2) Frame count distribution
3) Frame size distribution
4) TICI label distribution

Usage:
    uv run python generate_dataset_flowchart.py
    uv run python generate_dataset_flowchart.py --data-dir AmTICIS --output dataset_flowchart.png
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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
        default=Path("dataset_flowchart.png"),
        help="Output PNG path (default: dataset_flowchart.png).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="How many entries to show for frame counts/sizes in each node.",
    )
    return parser.parse_args()


def format_counter(counter: Counter, top_k: int) -> str:
    """Format a counter as multi-line text."""
    items = counter.most_common(top_k)
    lines = [f"{k}: {v}" for k, v in items]
    if len(counter) > top_k:
        rest = sum(v for _, v in counter.most_common()[top_k:])
        lines.append(f"... +{len(counter) - top_k} more ({rest} scans)")
    return "\n".join(lines) if lines else "No data"


def draw_node(ax, center_xy, width, height, text, fontsize=10):
    x, y = center_xy
    left = x - width / 2
    bottom = y - height / 2
    patch = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor="black",
        facecolor="#f7f9fc",
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, family="sans-serif")


def connect(ax, start_xy, end_xy):
    ax.annotate(
        "",
        xy=end_xy,
        xytext=start_xy,
        arrowprops=dict(arrowstyle="-", lw=1.4, color="black"),
    )


def scan_metadata(data_dir: Path):
    view_counts = Counter()
    frame_count_counts = Counter()
    frame_size_counts = Counter()
    label_counts = Counter()

    nii_files = sorted(data_dir.glob("*.nii.gz"))
    if not nii_files:
        raise ValueError(f"No .nii.gz files found in {data_dir}")

    for fpath in nii_files:
        fname = fpath.name

        # View from filename marker.
        vm = re.search(r"_(C|S)(?:#|_)", fname)
        if vm:
            view = "AP" if vm.group(1) == "C" else "sagittal"
            view_counts[view] += 1

        # TICI label from filename.
        parts = fname.split("_")
        if len(parts) >= 2:
            label = parts[1].upper()
            label_counts[label] += 1

        # Shape from NIfTI array: (H, W, T)
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(fpath)))
        h, w, t = map(int, arr.shape)
        frame_count_counts[t] += 1
        frame_size_counts[f"{h}x{w}"] += 1

    return len(nii_files), view_counts, frame_count_counts, frame_size_counts, label_counts


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    total_scans, view_counts, frame_counts, size_counts, label_counts = scan_metadata(
        args.data_dir
    )

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Root + 4 branches.
    root_xy = (0.5, 0.88)
    branches = {
        "view": (0.17, 0.50),
        "frames": (0.39, 0.50),
        "sizes": (0.61, 0.50),
        "labels": (0.83, 0.50),
    }

    root_text = f"AmTICIS Dataset\nTotal scans: {total_scans}"
    draw_node(ax, root_xy, 0.34, 0.12, root_text, fontsize=14)

    view_text = "Scans per view\n\n" + format_counter(view_counts, args.top_k)
    frame_text = "Frame counts\n(scans per frame count)\n\n" + format_counter(
        frame_counts, args.top_k
    )
    size_text = "Frame sizes (HxW)\n(scans per size)\n\n" + format_counter(
        size_counts, args.top_k
    )
    label_text = "TICI labels\n(scans per label)\n\n" + format_counter(label_counts, args.top_k)

    draw_node(ax, branches["view"], 0.28, 0.58, view_text, fontsize=10)
    draw_node(ax, branches["frames"], 0.28, 0.58, frame_text, fontsize=10)
    draw_node(ax, branches["sizes"], 0.28, 0.58, size_text, fontsize=10)
    draw_node(ax, branches["labels"], 0.28, 0.58, label_text, fontsize=10)

    for bxy in branches.values():
        connect(ax, (root_xy[0], root_xy[1] - 0.07), (bxy[0], bxy[1] + 0.30))

    plt.title("AmTICIS Scan Summary Flowchart", fontsize=18, pad=14)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=220)
    plt.close(fig)

    print(f"Saved flowchart: {args.output}")
    print(f"Total scans: {total_scans}")


if __name__ == "__main__":
    main()
