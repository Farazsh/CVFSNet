"""Build a Model x View comparison table (AUROC/AUPRC/accuracy/F1/recall/
precision/specificity) from the CVFSNet and VideoMAE binary metrics CSVs.

Recall, precision, and specificity are derived from the tp/tn/fp/fn columns
(not stored directly in the source CSVs); accuracy/F1/AUROC/AUPRC are taken
as-is from those CSVs. All values are rounded to 3 decimal places.

Usage:
    uv run python -m amticis_training.build_metrics_comparison_table

Output: Assets/metrics_comparison_table.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    ("CVFSNet", REPO_ROOT / "Assets" / "confusion_matrix_metrics_cvfsnet_binary_runs_mod3.csv"),
    ("VideoMAE", REPO_ROOT / "Assets" / "confusion_matrix_metrics_videomae_binary_runs.csv"),
]

# A source CSV may hold rows from several model families (the CVFSNet file also
# carries the MedGemma runs), so the display name comes from the row's own
# ``experiment`` column. The SOURCES label is only a fallback for rows that
# predate that column.
MODEL_NAMES = {
    "cvfsnet_binary": "CVFSNet",
    "videomae_binary": "VideoMAE",
    "medgemma_qlora_binary": "MedGemma QLoRA",
    "medgemma_zero_shot_binary": "MedGemma zero-shot",
}

COLUMNS = [
    "Model",
    "Views",
    "SPLIT",
    "N",
    "AUROC",
    "AUPRC",
    "ACCURACY",
    "F1",
    "RECALL",
    "PRECISION",
    "SPECIFICITY",
]


def _fmt3(value: float) -> str:
    return f"{float(value):.3f}"


def _row_for(model: str, row: dict) -> dict:
    tp, tn, fp, fn = int(row["tp"]), int(row["tn"]), int(row["fp"]), int(row["fn"])
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "Model": MODEL_NAMES.get(row.get("experiment", ""), model),
        "Views": row["title"],
        # SPLIT and N are surfaced because not every row is scored on the same
        # studies: the zero-shot rows are the 53-study tuning split, not the
        # 150-study val split, and must not be read as directly comparable.
        "SPLIT": row.get("split", "val"),
        "N": row["n"],
        "AUROC": _fmt3(row["auroc"]),
        "AUPRC": _fmt3(row["auprc"]),
        "ACCURACY": _fmt3(row["accuracy"]),
        "F1": _fmt3(row["f1_macro"]),
        "RECALL": _fmt3(recall),
        "PRECISION": _fmt3(precision),
        "SPECIFICITY": _fmt3(specificity),
    }


def build_rows() -> List[dict]:
    rows: List[dict] = []
    for model, csv_path in SOURCES:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for source_row in csv.DictReader(handle):
                rows.append(_row_for(model, source_row))
    return rows


def main() -> None:
    rows = build_rows()

    out = REPO_ROOT / "Assets" / "metrics_comparison_table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}")

    header = " | ".join(COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(str(row[col]) for col in COLUMNS))


if __name__ == "__main__":
    main()
