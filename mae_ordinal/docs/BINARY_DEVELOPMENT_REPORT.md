# VideoMAE Binary TICI Classification Development Report
_Exported on 7/11/2026 at 14:28:33 GMT+2 from Cursor (3.10.20)_

---

**User**

Here is an exported chat from previous analysis VideoMAE_ordinal_regression_tici_grading.md
Read this to get the idea of the repository and overall project strcuture and analysis so far.

Design a binary TICI classification experiment T012a vs T2b3, using the VideoMAE and binary cross entropy loss. Use the same data preprocessing and Normalization as mentioned in VideoMAE_ordinal_regression_tici_grading.md
Run 3 experiments in parallel, one with AP only, another with sagittal only and a third with boh views. 

Use different gpus for each exps and different terminals for each exps, so as to run all exp in parallel. After the training is finished perform an evaluation of the best model checkpoint and capture the sigmoid score of all scans into a csv file for a later analysis. Also use the confusion matrix plotting function present in the repo for comparing the results of baseline and the new exps. 

PLAN only and DONOT write code. Commit the existing changes and Switch to a new branch mae-binary-mix. 
make a plan md file named, VideoMAE_Binary_TICI_classification.md. If you have any unresolved questions, ask now and donot assume. 


---

**Cursor**

Reviewing the exported analysis and codebase to draft the binary TICI classification plan.


Checking the implementation for binary labels, dual-view handling, and evaluation patterns.


Checking class distribution for binary labels and whether CVFSNet view-specific baselines exist.


A few design choices need your input before I finalize the plan.


**Binary TICI experiment — clarifications**

> **For confusion-matrix comparison vs "baseline", which approach do you want? (There are no existing CVFSNet runs trained with label_mode=binary; only fuse01 4-class runs exist: cor_v1 / sag_v1 / fusion_v1.)**
> - Use existing CVFSNet fuse01 checkpoints (view-matched: cor/sag/fusion), collapse 4-class preds to binary (T0/1+T2a→0, T2b+T3→1) for CM comparison ✔
> - Train 3 new CVFSNet binary runs (AP / sagittal / both) with BCE, then compare CMs apples-to-apples
> - Compare only the 3 new VideoMAE binary runs (no CVFSNet baseline on the plot)
> - Other (I'll specify in chat)

> **For the "both views" VideoMAE experiment, how should AP + sagittal be fused?**
> - Shared VideoMAE backbone: encode AP and sagittal separately, concat pooled features → binary head (lighter, ~86M params) ✔
> - Two independent VideoMAE encoders (CVFSNet-style dual branch), concat features → binary head (~172M params)
> - Other (I'll specify in chat)

> **Training mode for all 3 VideoMAE runs? (Ordinal experiment showed full fine-tune clearly beats frozen.)**
> - Full fine-tune only (VideoMAE FINETUNE.md recipe: LLRD 0.75, lr 2e-4, wd 0.05, warmup+cosine) ✔
> - Frozen backbone only (linear probe)
> - Run both frozen and fine-tune for each view (6 runs total — not what you asked, but available)

Creating the plan with your decisions incorporated.


---

**User**

VideoMAE Binary TICI Classification (T012a vs T2b3)

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

Implementing the plan step by step, starting with git setup and reviewing the existing codebase.


Writing documentation while training runs, then monitoring until completion.


All three training runs finished. Running evaluation and plotting.


All tasks are complete on branch `mae-binary-mix`. Summary:

## Git
- Committed the ordinal experiment work on `mae-ordinal-coronal`
- Created and switched to **`mae-binary-mix`**

## Implementation
- **`VideoMAEBinary`** / **`VideoMAEBinaryDual`** (shared backbone, late fusion) with BCE head in [`mae_ordinal/model.py`](../model.py)
- Binary view presets in [`mae_ordinal/dataloader.py`](../dataloader.py) (`ap_binary`, `sagittal_binary`, `dual_binary`)
- [`mae_ordinal/binary_lightning_module.py`](../binary_lightning_module.py) with BCE + `ClassificationMetricAccumulator`
- Three configs: `config_binary_ap.yaml`, `config_binary_sagittal.yaml`, `config_binary_dual.yaml`
- [`mae_ordinal/eval_binary.py`](../eval_binary.py) — sigmoid CSV export
- [`mae_ordinal/plot_confusion_binary.py`](../plot_confusion_binary.py) — CM vs collapsed CVFSNet baselines

## Training (parallel on GPUs 0/1/2)
All three runs finished successfully.

| Run | Val accuracy | F1 | AUROC |
|-----|-------------:|---:|------:|
| AP | 0.827 | 0.826 | 0.925 |
| Sagittal | 0.780 | 0.779 | 0.845 |
| **Dual** | **0.853** | **0.853** | 0.895 |

## Outputs
- **Sigmoid CSVs:** `output_runs_lightning/mae_binary_t012a_{ap,sag,dual}/val_sigmoid_scores.csv`
- **Confusion matrices:** `plots/confusion_matrices_binary.png`
- **Docs:** [`BINARY_EXPERIMENT_PLAN.md`](BINARY_EXPERIMENT_PLAN.md), [`BINARY_RESULTS_REPORT.md`](BINARY_RESULTS_REPORT.md)

## vs collapsed CVFSNet fuse01 baselines
| View | VideoMAE | CVFSNet (collapsed) |
|------|---------:|--------------------:|
| AP | 0.827 | 0.813 |
| Sagittal | 0.780 | 0.767 |
| Dual | **0.853** | 0.813 |

Dual-view VideoMAE gives the largest gain (+4.0% acc over the fusion baseline). AP-only also beats its baseline with the highest AUROC (0.925).

Binary code changes are uncommitted on `mae-binary-mix` — say if you want a commit.

---

**User**

<system_notification>
The following task has finished. If you were already aware, ignore this notification and do not restate prior responses.

<task>
kind: shell
status: success
task_id: 346918
title: Launch sagittal binary training on GPU 1
tool_call_id: tool_cedc7684-dc0d-4007-b4a7-e7b096d5afa
output_path: /home/fshaikh/.cursor/projects/home-fshaikh-Projects-CVFSNet/terminals/346918.txt
</task>
</system_notification>
<user_query>Briefly inform the user about the task result and perform any follow-up actions (if needed). If there's no follow-ups needed, don't explicitly say that.</user_query>

<system_notification>
The following task has finished. If you were already aware, ignore this notification and do not restate prior responses.

<task>
kind: shell
status: success
task_id: 346917
title: Launch AP binary training on GPU 0
tool_call_id: tool_bccd864f-f2ad-459c-9494-db970148bf4
output_path: /home/fshaikh/.cursor/projects/home-fshaikh-Projects-CVFSNet/terminals/346917.txt
</task>
</system_notification>
<user_query>Briefly inform the user about the task result and perform any follow-up actions (if needed). If there's no follow-ups needed, don't explicitly say that.</user_query>

---

**Cursor**

Both parallel training runs finished successfully:

- **AP (`mae_binary_t012a_ap`, GPU 0):** val accuracy **0.827**, F1 **0.826**, AUROC **0.925**
- **Sagittal (`mae_binary_t012a_sag`, GPU 1):** val accuracy **0.780**, F1 **0.779**, AUROC **0.845**

Evaluation is already done — sigmoid scores are in `output_runs_lightning/mae_binary_t012a_{ap,sag}/val_sigmoid_scores.csv`, and the baseline comparison plot is at `plots/confusion_matrices_binary.png`. Full write-up is in [`BINARY_RESULTS_REPORT.md`](BINARY_RESULTS_REPORT.md).

---

**User**

Commit the changes to the git, exclude all large files like saved weight and csv files and plots. add them to git ignore. 

---

**Cursor**

Checking git state and `.gitignore` before committing and excluding large artifacts.


Updating `.gitignore` with explicit checkpoint/weight patterns, then committing code and docs only.


Committed on **`mae-binary-mix`** as `b5c8869`.

**Included:** binary experiment code (`mae_ordinal/`), configs, eval/plot scripts,
`mae_ordinal/docs/BINARY_RESULTS_REPORT.md`,
`mae_ordinal/docs/BINARY_EXPERIMENT_PLAN.md`, and updated `.gitignore`.

**Excluded via `.gitignore`:** `output_runs_lightning/` (~6 GB checkpoints/CSVs), `plots/`, `*.csv`, `*.png`, `*.ckpt`, `*.pt`, `*.pth`, `*.safetensors`, `*.bin`, `mae_*.log`, and W&B dirs.

**Still untracked:** `VideoMAE_ordinal_regression_tici_grading.md` (prior chat export — not part of this commit). Say if you want that added too.

---

**User**

push all branches and changes to the the remote

---

**Cursor**

Pushed to `origin` (`git@github.com:Farazsh/CVFSNet.git`):

| Branch | Status |
|--------|--------|
| **`mae-binary-mix`** | New on remote (binary T012a vs T2b3 experiment) |
| **`mae-ordinal-coronal`** | New on remote (ordinal regression + chat export) |
| **`main`** | Already up to date |
| **`dataset_structuring`** | Already up to date |

Large artifacts (checkpoints, CSVs, plots) stay local via `.gitignore` and were not pushed.
