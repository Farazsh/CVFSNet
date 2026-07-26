"""Generate a stacked bar plot of frame-count frequency for pretraining_DSA.

Modeled on ``generate_frame_count_barplot.py`` (AmTICIS view stacking) and the
DIAS / DSA_public variants. Reads flat NIfTI volumes in ``pretraining_DSA`` and
colors bars by source dataset (DSCA vs DIAS) using the metadata CSV/Excel file.

Usage:
    uv run python plotting_functions/generate_pretraining_dsa_frame_count_barplot.py
    uv run python plotting_functions/generate_pretraining_dsa_frame_count_barplot.py \\
        --data-dir pretraining_DSA \\
        --metadata pretraining_DSA/pretraining_DSA_metadata.csv \\
        --output plots/pretraining_dsa_frame_count_barplot.png
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib

# Palette slots 1 and 3 from the DIAS / DSA_public plots (blue / magenta).
DATASET_COLORS = {
    "DSCA": "#2a78d6",
    "DIAS": "#e87ba4",
}
DATASET_ORDER = ("DSCA", "DIAS")
# Excel/CSV stores DSCA scans as "DSA".
DATASET_ALIASES = {
    "DSA": "DSCA",
    "DSCA": "DSCA",
    "DIAS": "DIAS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("pretraining_DSA"),
        help="Directory of pretraining_DSA NIfTI scans (default: pretraining_DSA).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("pretraining_DSA/pretraining_DSA_metadata.csv"),
        help="CSV/Excel metadata with Dataset column "
        "(default: pretraining_DSA/pretraining_DSA_metadata.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/pretraining_dsa_frame_count_barplot.png"),
        help="Output plot file path.",
    )
    return parser.parse_args()


def load_filename_to_dataset(metadata_path: Path) -> dict[str, str]:
    if metadata_path.suffix.lower() == ".csv":
        with metadata_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        import pandas as pd

        rows = pd.read_excel(metadata_path).to_dict(orient="records")

    if not rows:
        raise ValueError(f"Metadata is empty: {metadata_path}")
    required = {"New_Filename", "Dataset"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")

    mapping: dict[str, str] = {}
    for row in rows:
        raw = str(row["Dataset"]).strip()
        if raw not in DATASET_ALIASES:
            raise ValueError(f"Unknown dataset label in metadata: {raw!r}")
        mapping[str(row["New_Filename"])] = DATASET_ALIASES[raw]
    return mapping


def get_frame_count(nii_path: Path) -> int:
    """Return the number of frames (3rd axis) without loading pixel data."""
    return int(nib.load(nii_path).shape[2])


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")

    filename_to_dataset = load_filename_to_dataset(args.metadata)
    nii_files = sorted(args.data_dir.glob("*.nii.gz"))
    if not nii_files:
        raise ValueError(f"No .nii.gz files found in: {args.data_dir}")

    frame_counts: dict[str, Counter[int]] = {d: Counter() for d in DATASET_ORDER}
    for nii_path in nii_files:
        dataset = filename_to_dataset.get(nii_path.name)
        if dataset is None:
            raise KeyError(f"No metadata row for file: {nii_path.name}")
        frame_counts[dataset][get_frame_count(nii_path)] += 1

    datasets = [d for d in DATASET_ORDER if sum(frame_counts[d].values()) > 0]
    x_vals = sorted(set().union(*(frame_counts[d].keys() for d in datasets)))
    per_dataset_vals = {d: [frame_counts[d][x] for x in x_vals] for d in datasets}

    plt.figure(figsize=(14, 6))
    bottoms = [0] * len(x_vals)
    for dataset in datasets:
        vals = per_dataset_vals[dataset]
        plt.bar(
            x_vals,
            vals,
            bottom=bottoms,
            color=DATASET_COLORS[dataset],
            edgecolor="black",
            linewidth=0.5,
            label=dataset,
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    plt.title("Frame Count Distribution Across Pretraining DSA Scans")
    plt.xlabel("Number of Frames per Scan")
    plt.ylabel("Number of Scans")
    plt.xticks(x_vals, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(title="Dataset")

    for i, total in enumerate(bottoms):
        if total > 0:
            plt.text(x_vals[i], total + 0.5, str(total), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    plt.close()

    print(f"Saved bar plot to: {args.output}")
    print(f"Total scans processed: {len(nii_files)}")
    for dataset in datasets:
        print(f"  {dataset}: {sum(per_dataset_vals[dataset])}")
    print(f"Unique frame counts: {len(x_vals)}")


if __name__ == "__main__":
    main()
