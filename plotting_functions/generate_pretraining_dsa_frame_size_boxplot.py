"""Plot frame-size boxplots for pretraining_DSA, colored by DSCA vs DIAS.

Modeled on ``plot_frame_size_counts.py`` (AmTICIS frame-size extraction), but
shows Height and Width distributions as boxplots instead of stacked size-count
bars. Source dataset comes from the pretraining_DSA metadata CSV/Excel.

Usage:
    uv run python plotting_functions/generate_pretraining_dsa_frame_size_boxplot.py
    uv run python plotting_functions/generate_pretraining_dsa_frame_size_boxplot.py \\
        --data-dir pretraining_DSA \\
        --metadata pretraining_DSA/pretraining_DSA_metadata.csv \\
        --output plots/pretraining_dsa_frame_size_boxplot.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib

DATASET_COLORS = {
    "DSCA": "#2a78d6",
    "DIAS": "#e87ba4",
}
DATASET_ORDER = ("DSCA", "DIAS")
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
        help="CSV/Excel metadata with Dataset column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/pretraining_dsa_frame_size_boxplot.png"),
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


def get_spatial_size(nii_path: Path) -> tuple[int, int]:
    """Return spatial frame size (H, W) without loading pixel data."""
    shape = nib.load(nii_path).shape
    return int(shape[0]), int(shape[1])


def collect_sizes(
    data_dir: Path, filename_to_dataset: dict[str, str]
) -> dict[str, dict[str, list[int]]]:
    """Return {dataset: {"H": [...], "W": [...]}}."""
    sizes: dict[str, dict[str, list[int]]] = {
        d: {"H": [], "W": []} for d in DATASET_ORDER
    }
    nii_files = sorted(data_dir.glob("*.nii.gz"))
    if not nii_files:
        raise ValueError(f"No .nii.gz files found in: {data_dir}")

    for nii_path in nii_files:
        dataset = filename_to_dataset.get(nii_path.name)
        if dataset is None:
            raise KeyError(f"No metadata row for file: {nii_path.name}")
        h, w = get_spatial_size(nii_path)
        sizes[dataset]["H"].append(h)
        sizes[dataset]["W"].append(w)
    return sizes


def plot_boxplots(sizes: dict[str, dict[str, list[int]]], output_path: Path) -> None:
    datasets = [d for d in DATASET_ORDER if sizes[d]["H"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    for ax, dim in zip(axes, ("H", "W")):
        data = [sizes[d][dim] for d in datasets]
        box = ax.boxplot(
            data,
            tick_labels=datasets,
            patch_artist=True,
            widths=0.55,
            showfliers=True,
        )
        for patch, dataset in zip(box["boxes"], datasets):
            patch.set_facecolor(DATASET_COLORS[dataset])
            patch.set_alpha(0.85)
            patch.set_edgecolor("black")
        for key in ("whiskers", "caps"):
            for line in box[key]:
                line.set_color("black")
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        ax.set_title(f"Frame {dim}")
        ax.set_xlabel("Dataset")
        ax.set_ylabel(f"{dim} (pixels)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(
        "Frame Size Distribution Across Pretraining DSA Scans",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")

    filename_to_dataset = load_filename_to_dataset(args.metadata)
    sizes = collect_sizes(args.data_dir, filename_to_dataset)
    plot_boxplots(sizes, args.output)

    total = sum(len(sizes[d]["H"]) for d in DATASET_ORDER)
    print(f"Saved plot to: {args.output}")
    print(f"Total scans processed: {total}")
    for dataset in DATASET_ORDER:
        n = len(sizes[dataset]["H"])
        if n == 0:
            continue
        h_vals = sizes[dataset]["H"]
        w_vals = sizes[dataset]["W"]
        print(
            f"  {dataset}: n={n}, "
            f"H=[{min(h_vals)}, {max(h_vals)}], "
            f"W=[{min(w_vals)}, {max(w_vals)}]"
        )


if __name__ == "__main__":
    main()
