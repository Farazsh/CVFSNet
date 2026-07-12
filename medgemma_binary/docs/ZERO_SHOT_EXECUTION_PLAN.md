# MedGemma Zero-Shot Binary TICI Execution Plan

The detailed experiment plan is kept in
[`BINARY_TICI_EXPERIMENT_PLAN.md`](BINARY_TICI_EXPERIMENT_PLAN.md).

The original run is invalid because every class score was non-finite and was
silently replaced with `-1e9`. Its accuracy, F1, AUROC, and confusion matrix
must not be interpreted as MedGemma performance.

Create an isolated environment for the corrected runners; do not apply the old
Torch attention-mask patch:

```text
uv venv .venv-medgemma --python 3.12
uv pip install --python .venv-medgemma/bin/python -r medgemma_binary/requirements-medgemma.txt
```

Run a balanced smoke evaluation, tune the predefined prompt/scaling grid on
the internal tuning split, and only then run the locked final endpoint:

```text
.venv-medgemma/bin/python -m medgemma_binary.zero_shot --phase smoke
.venv-medgemma/bin/python -m medgemma_binary.zero_shot --phase tuning
.venv-medgemma/bin/python -m medgemma_binary.zero_shot --phase final
```

The final phase requires the `selection.yaml` written by the tuning phase. The
runner uses eight chronological frames per AP and sagittal view at MedGemma's
native 896-pixel input, BF16 inference, and one next-token forward pass. Any
non-finite score aborts the run.

Run the MedSigLIP zero-shot control with the same phase sequence:

```text
.venv-medgemma/bin/python -m medgemma_binary.medsiglip_zero_shot --phase smoke
.venv-medgemma/bin/python -m medgemma_binary.medsiglip_zero_shot --phase tuning
.venv-medgemma/bin/python -m medgemma_binary.medsiglip_zero_shot --phase final
```
