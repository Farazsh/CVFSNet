# Results — VideoMAE Binary T012a vs T2b3

Three parallel full-fine-tune runs: AP-only, sagittal-only, and dual-view (shared-backbone
late fusion). Binary cross-entropy loss, ImageNet adapter normalization, same train/val split
(261/150) as prior experiments.

W&B project: `AmTICIS`

## Runs

| Run | View | Config | GPU |
|-----|------|--------|-----|
| `mae_binary_t012a_ap` | AP (coronal) | `mae_ordinal/config_binary_ap.yaml` | 0 |
| `mae_binary_t012a_sag` | Sagittal | `mae_ordinal/config_binary_sagittal.yaml` | 1 |
| `mae_binary_t012a_dual` | AP + sagittal | `mae_ordinal/config_binary_dual.yaml` | 2 |

## VideoMAE results (best checkpoint, val set)

| Run | Accuracy | F1 macro | AUROC |
|-----|---------:|---------:|------:|
| AP | **0.827** | **0.826** | **0.925** |
| Sagittal | 0.780 | 0.779 | 0.845 |
| Dual | **0.853** | **0.853** | 0.895 |

Sigmoid score CSVs (150 val scans each):
- `output_runs_lightning/mae_binary_t012a_ap/val_sigmoid_scores.csv`
- `output_runs_lightning/mae_binary_t012a_sag/val_sigmoid_scores.csv`
- `output_runs_lightning/mae_binary_t012a_dual/val_sigmoid_scores.csv`

Columns: `name`, `true_label`, `sigmoid_score` (P(T2b3)), `pred_label`.

## Baseline comparison (collapsed fuse01 → binary)

CVFSNet fuse01 4-class predictions collapsed: classes 0,1 → T012a (0); 2,3 → T2b3 (1).
These baselines were **not** trained with binary BCE — collapse is post-hoc.

| View | VideoMAE acc | CVFSNet collapsed acc | Δ |
|------|-------------:|----------------------:|--:|
| AP | 0.827 | 0.813 | +0.014 |
| Sagittal | 0.780 | 0.767 | +0.013 |
| Dual | **0.853** | 0.813 | **+0.040** |

Confusion matrices: `plots/confusion_matrices_binary.png` (3×2 grid: VideoMAE vs collapsed CVFSNet per view).

## Summary

- **Dual-view VideoMAE is the strongest** binary classifier (acc 0.853, F1 0.853), beating the
  collapsed CVFSNet fusion baseline by ~4 points accuracy.
- **AP-only VideoMAE** also edges the coronal CVFSNet baseline (+1.4% acc) with the highest AUROC
  (0.925).
- **Sagittal-only** is weakest among the three VideoMAE runs but still slightly above its collapsed
  CVFSNet sagittal baseline.
- Combining AP + sagittal via shared-backbone late fusion gives a clear gain over either single view
  (+2.6% vs AP, +7.3% vs sagittal on accuracy).

## Caveats

- `test = val` (no held-out test set); best-epoch-on-val selection.
- CVFSNet baselines are fuse01 4-class models collapsed to binary, not natively BCE-trained.
- Single seed; margins on a 150-sample val set should be treated as indicative, not conclusive.

## Reproduce

```bash
uv run python -m mae_ordinal.eval_binary
uv run python -m mae_ordinal.plot_confusion_binary
```
