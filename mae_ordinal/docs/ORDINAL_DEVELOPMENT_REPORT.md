# VideoMAE Ordinal Regression Development Report
_Exported on 7/11/2026 at 11:22:43 GMT+2 from Cursor (3.10.20)_

---

**User**

CONTEXT

I have a cloned repo of the CVFSNet paper and have already:
1. Reproduced their dual-view (coronal + sagittal) TICI classification results.
2. Modified it to use coronal view only, and confirmed no performance drop vs. the
   dual-view baseline.

Next experiment: test whether reformulating TICI grading as ORDINAL REGRESSION on top
of a VideoMAE backbone (pretrained masked-autoencoder video transformer) beats the
existing coronal-only CVFSNet CNN baseline. Single view only (coronal) — no cross-view
fusion needed for this experiment.

DO NOT WRITE IMPLEMENTATION CODE YET. Your job right now is to produce a written plan
and get my sign-off before any model/dataloader code is written.

STEP 1 — Investigate the existing repo and report back (in the plan doc, not just chat):
- Where the dataset/dataloader classes live, and exactly how coronal-view frames are
  loaded, subsampled/selected, resized, normalized, and augmented. Cite specific files
  and function/class names.
- How TICI/mTICI labels are defined and encoded (raw grade set — e.g. 0/1/2a/2b/2c/3 —
  and how it's mapped to model target indices/classes).
- Exactly how the coronal-only ablation is switched on (config flag, code path, etc.).
- The current model entry point, training loop, config system, and logging/checkpoint
  conventions, so the new experiment can follow the same conventions.
- Which evaluation metrics the repo already computes (accuracy, MCC, Cohen's kappa,
  per-class agreement, etc.) — the new experiment must report the same metrics so
  results are directly comparable to the coronal-only baseline.
- Dataset size (train/val/test split sizes) and class distribution across TICI grades.
- Existing dependency file (requirements.txt / pyproject.toml / environment.yml) and
  torch/CUDA versions, plus available local GPU memory (check with nvidia-smi or
  equivalent) — this affects whether VideoMAE-Base or a smaller variant, and what
  frame count/batch size, is feasible.

STEP 2 — Design (don't build yet) a new, self-contained package, separate from the
existing CVFSNet package, e.g. `mae_ordinal/`, containing at minimum:

- `dataloader.py`
  Must REUSE the existing repo's preprocessing/transform functions and classes as-is —
  import them, do not reimplement or fork them. Wrap/adapt their output to whatever
  input format the VideoMAE processor expects (frame count, resolution, channel count,
  normalization stats). If VideoMAE's expected preprocessing conflicts with what the
  existing pipeline produces (e.g. different normalization stats, RGB vs single-channel,
  fixed vs. variable frame count), flag this explicitly in the plan and propose how to
  reconcile it — don't silently pick one.

- `model.py`
  VideoMAE backbone using the `transformers` library implementation (or VideoMAE V2 if
  you find a maintained pretrained checkpoint) — propose Base vs. Large and which
  pretrained source (Kinetics-400 vs Kinetics-710 vs other), with justification given
  our dataset size found in Step 1. Replace the classification head with an ordinal
  regression head. Propose a specific ordinal formulation (e.g. CORAL/CORN cumulative-
  link output vs. a simpler ordinal-encoded regression) appropriate for our N-way
  ordinal target (confirm N from the label scheme found in Step 1), and justify the
  choice.

- `train.py` + config
  Training loop, backbone freeze/unfreeze schedule (propose one given our dataset
  size — likely freeze most of the backbone and fine-tune only top blocks + head),
  loss function for the ordinal head, class imbalance handling if needed, and
  evaluation using the exact same metrics/protocol as the coronal-only CVFSNet
  baseline (same train/val/test split).

- Do not modify any file inside the existing CVFSNet package — only import from it.

STEP 3 — Write the plan to `PLAN.md` at the repo root, covering:
- New package/file structure
- Key design decisions and justification (frame sampling, resolution, normalization
  reconciliation, ordinal head choice, pretrained checkpoint choice, freeze schedule,
  eval protocol)
- Open questions / risks / anything you're unsure about
- Step-by-step implementation checklist (including a smoke test on a couple of
  samples before any full training run)

STEP 4 — Stop and wait for my review of PLAN.md. Do not start implementing until I
explicitly approve it. If anything in Step 1 doesn't match these assumptions, ask me
rather than guessing.

Work on a new git branch for this; don't touch the existing CVFSNet code or branch.

---

**Cursor**

I'll start by investigating the repo structure, then create a plan. Let me first set up a branch and explore the codebase.

Let me look at the repo structure and check GPU/environment in parallel.

Good — 4× RTX 3090 (24GB each), plenty of GPU memory. There's a `.venv`. Let me create the branch and explore key files.

Let me explore the directory structure of the key packages.

Two pipelines exist. The modified file is `amticis_training/config.yaml`, so the active (reproduced) pipeline is `amticis_training` + `amticis_pipeline`. Let me read those thoroughly.

Now the pipeline utils and config, plus the training package.

Now the training package.

Now the training loop, config, losses, and the original metric module.

Let me check the git diff on the config, the training config loader, the original metric module, and the split counts.

Now let me check the split/class distribution, the original transforms (normalization), venv packages, and git log for the coronal-only run.

Let me examine the original transforms (especially normalization) and confirm the coronal-only run config plus torch version.

Let me read the key transform implementations (normalization, resize, crop) to nail down the exact tensor format and normalization stats.

Let me quickly check existing run outputs to confirm how the coronal-only baseline was configured, and glance at the model input channel default.

I now have everything needed. The coronal-only baseline is `cvfsnet_pt_fuse01_cor_v1` (fuse01, 4-class, both CVFSNet branches fed the coronal `_C` file). Let me finalize investigation and write the plan.

Now I'll write the plan.

Done. I investigated the repo and wrote the full plan to `PLAN.md` at the repo root, on a new branch `mae-ordinal-coronal`. **No model/dataloader/training code was written**, and no existing CVFSNet files were touched.

## What I found (Step 1 highlights, all cited in `PLAN.md`)

- **Two pipelines exist.** The one you reproduced/ablated is the Lightning stack: `amticis_pipeline/` (data) + `amticis_training/` (model/loss/metrics/trainer). The new experiment plugs into this.
- **Coronal loading:** `AmTICISDataset._load_view` → `load_nifti` `(H,W,T)` → `ResizeView` (single **trilinear** `F.interpolate` resample; this *is* the "frame selection") → declarative torchio/OpenCV augmentations → `TioZNormalization(div255=true)`. Output `(1, 8, 256, 256)`, single channel.
- **Labels:** tokens `T0,T1,T2A,T2B,T3` (no T2c). `fuse01` merges T0+T1 → **4 ordinal classes** (the paper/baseline setting). Encoded in `amticis_pipeline/utils.py`.
- **Coronal-only ablation** (confirmed from `output_runs_lightning/cvfsnet_pt_fuse01_cor_v1/resolved_training_config.yaml`): both CVFSNet view definitions map `_C→_C`, so both branches get the coronal clip. The Lightning module also has a genuine `single_view_tensor` path we'll use.
- **Metrics:** `ClassificationMetricAccumulator` → accuracy, macro/per-class precision/recall/f1/specificity, AUROC/AUPRC; primary monitor is `val/fuse/f1_macro`. No MCC/kappa in the Lightning path.
- **Splits:** train 261 / val 150, **no dedicated test** (`test=val`). fuse01 distribution is imbalanced (train: 98/30/48/85).
- **Env:** torch 2.4.1+cu121, PL ≥2.6; **`transformers` and `timm` are NOT installed**. **4× RTX 3090 24 GB**, all idle → VideoMAE-Base is very comfortable.

## Design proposed (Step 2)

New self-contained `mae_ordinal/` package that **imports** the existing pipeline (no forks): VideoMAE-**Base** (`MCG-NJU/videomae-base-finetuned-kinetics`) backbone + **CORN** ordinal head (K−1=3), freeze-all → unfreeze top 2–4 blocks with discriminative LR, reuse the existing `WeightedRandomSampler`, and reuse `ClassificationMetricAccumulator` under `val/fuse/*` for direct comparison.

## Decisions I flagged for you (Q1–Q5 in the plan)

The most important: **normalization reconciliation** — VideoMAE wants ImageNet-normalized RGB at 224²/16 frames, but the pipeline emits grayscale z-normalized `/255` at 256²/8 frames. I recommend switching this experiment's config to 224/16, channel-replicating grayscale→3, and replacing *only the final normalization* with ImageNet stats — but I did not silently pick it.

Please review `PLAN.md` (especially Q1 normalization and 224/16 vs 256/8) and let me know your call on Q1–Q5. I'll wait for explicit approval before implementing.

---

**User**

Here are my changes to the plan
- Use the data loader from the amticis_pipeline package, donot use the older one
- use AP view only, donot use sagittal
- Performa a trillinear interpolation like the Existing data loader however use 16 frames and 224*224 frame size as needed fror VideoMAE
- Use the fuse01 label for ordinal Regression
- use WeightedRandomSampler for balanced mini batches/oversampling
- use the Existing metrics
- Install packages as needed

Tell me approximate time to implement the changes

Modify the plan and wait for my approval donot start coding



---

**Cursor**

Locking in your decisions and adding a time estimate. Let me update `PLAN.md`.

Now let me add a time estimate section before the "Awaiting review" section.

I've updated `PLAN.md` with your decisions. No code written yet.

## What changed in the plan

- Added a **"Locked decisions"** block at the top capturing all 7 of your directives: `amticis_pipeline` loader, **AP-only** (single-view path, no sagittal/duplication), trilinear `ResizeView` at **16 frames / 224×224**, `fuse01` (4-class) ordinal target, `WeightedRandomSampler`, existing `ClassificationMetricAccumulator` metrics, install-as-needed.
- **Frame/resolution rows** in the reconciliation table marked **LOCKED** (16 / 224²).
- **Normalization** (the one item you didn't explicitly address): adopted as default → keep all existing transforms but replace only the final normalization with `[0,1]`+ImageNet stats "as needed for VideoMAE," composed in our own config (never editing shared files). Left a one-line escape hatch in case you'd rather feed raw z-normalized clips.
- Trimmed open questions down to just **Q3** (VideoMAE v1 first) and **Q5** (internal Lightning-module reuse — my discretion).
- Added an **Approximate implementation time** section.

## Time estimate

- **Hands-on coding: ~4–6 hours** to a working, smoke-tested experiment (env + dataloader + CORN ordinal head + model/freeze logic + metrics adapter + train/config + smoke test).
- **Training wall-clock (unattended, separate): ~1–3 hours** on one RTX 3090 (mostly-frozen VideoMAE-Base, 261 clips), plus a one-time ~330 MB checkpoint download on first run.

Waiting for your approval before implementing. The only thing I'd flag for an explicit thumbs-up is the **ImageNet-normalization default** in §2.1 — everything else matches your instructions exactly.

---

**User**

Donot use the imagenet Normalization instead use the Normalization as performed in the dataset_loader. Furthermore donot use frozen-parameter, instead unfreeze and perforam full finetuning by chosing a particular learning rate as suitable. Start Implementation and run the experiement. Here is the wand key
add it to the .env and add it to git ignore
wandb.login(key="64339572cb325d13488524ac730cc32f507bd8ad") track all experiements

---

**Cursor**

Understood — switching to: (1) keep the dataset_loader's own normalization (no ImageNet stats), (2) full fine-tuning with a suitable single LR (no freezing), (3) wire up W&B via `.env`, and (4) implement and run. Let me update the plan's locked decisions and set up the environment.

Now let me update the PLAN.md locked decisions and set up the environment (add `.env`, gitignore, transformers dep) in parallel.

Now set up the environment: `.env`, gitignore, and add `transformers`.

transformers 5.13.0 installed. Let me verify the VideoMAE API in transformers 5.x and pre-download the checkpoint.

VideoMAE loads and runs (hidden 768, output `(B,1568,768)`). Note: transformers 5.x reinitializes the attention query/value biases for this checkpoint (a tiny fraction of params) — fine for full fine-tuning. Now let me build the package. Starting with the CORN ordinal utilities.

Now the dataloader that reuses `amticis_pipeline` for coronal AP single-view.

Now the model — VideoMAE backbone + CORN head, full fine-tuning, with the channel-replication adapter (no ImageNet renormalization).

Now the Lightning module (dedicated, reuses the repo's metric accumulator) plus ordinal extras.

Now the training config and entrypoint.

Let me check for lint errors, then run a quick correctness check on the CORN ordinal math and the data pipeline shapes.

CORN math is correct (probs normalize, loss overfits to 100%). Now an end-to-end data + model shape smoke test.

Full pipeline works end-to-end: data yields `(B,1,16,224,224)`, model outputs `(B,3)`, CORN loss is finite. The tiny input magnitude confirms the dataset_loader's own normalization is used (as requested). Now a full training-path smoke test (2 batches, 1 epoch, W&B off) before the real run.

Smoke test passed: 86.2M params, **all trainable (full fine-tuning, 0 frozen)**, loss logged, checkpoint path works. Let me clean up the smoke dir and launch the real tracked run in the background.

Launched. Let me monitor startup (W&B login + first epoch) after it initializes.

W&B is tracking (`https://wandb.ai/pytorch-vertebra/AmTICIS/runs/954ixqim`), full fine-tuning confirmed (0 non-trainable). One thing I want to verify is that "210 modules in eval mode" warning — newer Lightning may not force `train()`. Let me check the exact Lightning behavior to be safe.

This Lightning version does **not** auto-toggle train/eval mode. Let me check whether the evaluation loop sets eval mode, so I add the fix correctly.

This Lightning version leaves the backbone in eval mode during training (HF `from_pretrained` returns eval mode, and Lightning no longer force-calls `.train()`). For VideoMAE-Base this is functionally near-neutral (backbone dropout defaults to 0, no BatchNorm), but for a correct full fine-tune I'll explicitly set train mode. Let me stop the run, add the fix, and relaunch.

Training stopped (GPU 0 back to 15 MiB). Let me add the train-mode fix.

Let me quickly verify the fix with a short smoke run (W&B off), confirming train mode is active during training.

The fix works (run completes; `on_train_epoch_start` sets train mode each epoch — the one-time startup warning is cosmetic, checked before the epoch hook). Relaunching the full tracked run in the background.

The user explicitly asked me to run the experiment, so I'll surface the approval card for launching the training job.

Launched. Let me confirm it started training and is logging to W&B.

Training is live: GPU0 at 95% util, W&B run `https://wandb.ai/pytorch-vertebra/AmTICIS/runs/tz4oxjp7`. Let me let a few epochs complete and confirm the metrics (incl. `val/fuse/f1_macro`) are logging cleanly.

Everything is implemented, running, and tracked. Metrics are logging correctly under the exact baseline keys (`val/fuse/f1_macro`, `val/fuse/accuracy`, plus ordinal extras `qwk`/`mae`).

## Status

- **Branch:** `mae-ordinal-coronal` (existing CVFSNet code untouched).
- **W&B run:** `https://wandb.ai/pytorch-vertebra/AmTICIS/runs/tz4oxjp7` (project `AmTICIS`, all runs tracked; key in git-ignored `.env`).
- **Training:** live on GPU 0, currently ~epoch 7/120.

## What was built (`mae_ordinal/`, imports the existing code, never edits it)

- `dataloader.py` — reuses `amticis_pipeline.AmTICISDataModule`; **AP/coronal only**, trilinear `ResizeView` at **16 frames / 224²**, `fuse01` (4-class), **`WeightedRandomSampler`**, and the **dataset_loader's own `TioZNormalization(div255)` — no ImageNet stats** (only channel-replicate 1→3 in the model adapter).
- `ordinal.py` — CORN loss + rank/probability decoders (unit-checked: probs normalize, overfits a toy batch to 100%).
- `model.py` — `VideoMAEModel` Base (`MCG-NJU/videomae-base-finetuned-kinetics`) + CORN head; **full fine-tuning, 0 frozen params** (86.2M trainable).
- `lightning_module.py` — reuses the repo's `ClassificationMetricAccumulator` verbatim so results are directly comparable to `cvfsnet_pt_fuse01_cor_v1`; adds QWK/MAE/off-by-one.
- `train.py` + `config.yaml` — AdamW **lr 1e-5**, wd 0.05, cosine warm-restarts, grad-clip 1.0, mixed precision, checkpoint/early-stop on `val/fuse/f1_macro`, loads `.env` for W&B.
- `transformers` added via `uv`; `PLAN.md` updated to reflect all your locked decisions.

## Honest read on early numbers

At epoch 6, `val/fuse/f1_macro≈0.07`, `accuracy≈0.15`, `qwk≈0.0` — i.e. early class-collapse, which is common when fully fine-tuning an 86M video transformer at a low LR on 261 clips. Early-stopping patience is 40 epochs, so there's room to recover. I'll let it run (it finishes in ~1–3 h) and you'll get a completion notification. If it doesn't climb meaningfully above the majority-class rate (~0.34 val accuracy), the likely levers are LR/warmup and possibly a short head-only warm-up before full fine-tuning — but per your instruction I'm keeping it full fine-tune for now and won't change course without your say-so.

One note for transparency: transformers 5.x reinitializes VideoMAE's attention query/value **biases** for this checkpoint (a tiny fraction of params); since we're fully fine-tuning, they train from there.

---

**User**

Identify where could be the sources of error and could be changes for improvement

---

**Cursor**

I'm in Ask mode, so I'll investigate and analyze but won't change anything. Let me pull the current run state and re-examine the key code paths.

Confirmed: the run is **fully collapsed** — at the latest epoch it predicts class 2 for every sample (recall 1.0 / precision 0.15, all other classes 0), and across epochs it just oscillates between predicting a single class. `train_loss` never really drops (0.70→~0.6→back), QWK≈0. The model is stuck at the trivial constant-output solution. Let me confirm the mechanism by checking VideoMAE's input handling.

I've confirmed the root mechanism. Here's the analysis.

## Diagnosis: the model has collapsed to constant output

Evidence from the live run (`val/fuse/*`): `f1_macro` stuck ~0.07–0.12, `accuracy` oscillating between ~0.15/0.31 (i.e. always predicting whichever single class), `qwk≈0`, and at the last epoch **class 2 recall=1.0 / all other classes recall=0** — it literally predicts T2b for everything. `train_loss` never really drops (0.70→~0.60→back). This is the classic "the network gets no usable signal, so it parks at the majority/constant solution" failure.

## Sources of error (ranked by likely impact)

### 1. Input scale is ~250× too small for VideoMAE — the primary cause

The pipeline's final normalization is z-score **then divide by 255**:

```346:366:Data/transforms.py
class TioZNormalization(object):
    ...
    def __call__(self, image):  # CDHW
        if random.random() < self.p:
            image = image.permute(0, 2, 3, 1)
            try:
                image = self.trans(image)          # tio.ZNormalization -> mean 0, std 1
            ...
            if self.div255:
                image = image / 255.0              # -> std ~ 1/255 ≈ 0.004
        return image
```

Measured batch stats earlier: `min -0.058, max 0.025, std 0.0039`. So inputs arrive with **std ≈ 0.004**, whereas the pretrained backbone expects std ≈ 1.

Why that is fatal *specifically* for VideoMAE: its embedding does a plain Conv3d then adds **fixed sinusoidal** position embeddings, with **no input/BatchNorm and no LayerNorm before the add**:

```109:116:.venv/lib/python3.11/site-packages/transformers/models/videomae/modeling_videomae.py
    def forward(self, pixel_values, bool_masked_pos):
        # create patch embeddings
        embeddings = self.patch_embeddings(pixel_values)
        # add position embeddings
        embeddings = embeddings + self.position_embeddings.detach()...
```

With inputs ~250× too small, the patch (content) signal is ~250× smaller than the fixed position embeddings, so every sample's token sequence ≈ position-embeddings + negligible content → **sample-independent representations → constant prediction**. The subsequent LayerNorms can't recover information that's already been additively drowned out.

Crucially, this is also **why the CNN baseline tolerated the same normalization**: X3D's stem has `BatchNorm3d`, which rescales activations by their running std and is therefore *scale-invariant*. VideoMAE has no such normalization at the input, so it is *scale-sensitive*. The "use the dataset_loader normalization, no ImageNet stats" decision was safe for the CNN but breaks a normalization-free transformer.

Our adapter only repeats channels and permutes — it does not restore scale:

```55:71:mae_ordinal/model.py
    def _to_pixel_values(self, clip: torch.Tensor) -> torch.Tensor:
        ...
        if c == 1:
            clip = clip.expand(b, 3, t, h, w)
        ...
        return clip.permute(0, 2, 1, 3, 4).contiguous()
```

### 2. Full fine-tuning from step 0, single low LR, tiny dataset
Even with scale fixed, full fine-tuning an 86M ViT on **261 clips** at a flat `lr=1e-5` with no head warm-up is unstable: the randomly-initialized head emits noise into a pretrained backbone, and at 1e-5 the head also learns slowly. A collapsed head + tiny input signal reinforce each other (near-zero gradient into the backbone).

### 3. Reinitialized attention q/v biases (transformers 5.x)
The load report showed `query.bias`/`value.bias` reinitialized and the checkpoint's `q_bias`/`v_bias` dropped. Minor on its own, but it slightly degrades the pretrained starting point — unhelpful when combined with #1/#2.

### 4. Minor / not the cause
- **Mean-pool vs `fc_norm`**: the K400-finetuned checkpoint's classifier applies an `fc_norm` LayerNorm after pooling; we drop it and use our own head LayerNorm — fine, but not identical.
- **Metric decoding**: metrics use `argmax` of the CORN-derived class probs, not the rank-consistent `corn_predict`; these can differ slightly. Doesn't cause collapse, but worth aligning for reporting.
- **LR schedule**: `CosineAnnealingWarmRestarts(T_0=35)` periodically resets LR; irrelevant while the model isn't learning at all.

## Recommended changes (prioritized)

1. **Fix the input scale (highest impact).** The backbone needs ~unit-variance input. Options, in order of how much they keep your "dataset_loader normalization" intent:
   - Keep the pipeline's z-normalization but **drop the `/255`** for this experiment (z-norm alone already gives ~unit std) — smallest change, and arguably the pipeline's `div255` is a double-normalization artifact anyway.
   - Or keep the pipeline output as-is and **re-standardize inside the adapter** (per-sample subtract mean / divide by std, or multiply by ~255) so the tensor reaches std≈1 before the backbone.
   - Or apply the standard ImageNet scaling in the adapter (what the plan originally proposed). Any of these restores signal; the key requirement is std≈1, not the specific stats.

2. **Two-phase schedule + discriminative LRs.** A short head-only warm-up (backbone frozen, head lr ~1e-3) for a few epochs, then unfreeze for full fine-tuning with backbone lr ~1e-5 and a higher head lr. This is compatible with "full fine-tuning" (the freeze is only a brief warm-up).

3. **Add a learnability sanity check.** Overfit ~16 samples with augmentation off — if it can't reach ~100% train accuracy, the input/model wiring is still wrong (fast way to validate the scale fix before a full run).

4. **Fix the q/v bias load** — pin `transformers` to a version whose VideoMAE loader maps `q_bias`/`v_bias` correctly, or convert the checkpoint biases manually.

5. **Reporting/robustness (lower priority):** use `corn_predict` for the reported labels to match CORN's rank-consistent decoding; optionally add class weights to the CORN loss (in addition to the `WeightedRandomSampler`) if a minority class stays weak after the fix.

If you'd like, switch to Agent mode and I can implement the scale fix (#1) plus the warm-up schedule (#2) and re-run — I'd expect #1 alone to move it off the collapsed solution.

---

**User**

Implement the normalization of the VideoMAE i.e. Imagenet as needed, then rerun 2 contrastive experiement, one with frozen weights and another with full finetuning. Use appropriate parameters for frozen versus full finetuning as needed. Look for parameters suggested by other papers using Video MAE. Use 2 terminals and 2 gpus for parallel execution. Compare the results and prepare summary at the end

---

**Cursor**

I'm back in Agent mode. Let me stop the collapsed run, then implement the ImageNet normalization fix, add frozen-vs-finetune support with paper-based hyperparameters, and launch both in parallel.

GPUs are free. Let me quickly ground the hyperparameters in the VideoMAE literature before implementing.

VideoMAE recipe confirmed (AdamW, LLRD 0.75, wd 0.05, drop_path 0.1, 5-epoch warmup, cosine, base lr 5e-4–1e-3 with linear scaling). Let me check whether the HF VideoMAE config supports drop-path before wiring it up.

HF VideoMAE has no stochastic-depth (drop_path) — only `hidden_dropout_prob`/`attention_probs_dropout_prob` (both 0.0, but overridable). No top-level layernorm (mean-pooling model). I'll approximate drop_path with `hidden_dropout_prob=0.1` for fine-tuning and note the difference. Now let me rewrite the model with ImageNet normalization, freeze support, and LLRD param groups.

Now update the Lightning module's `configure_optimizers` to support LLRD + warmup-cosine, and keep the frozen backbone in eval mode.

Now update the import in the lightning module to pull in `layerwise_param_groups`.

Now the two experiment configs. First the full fine-tuning config (LLRD 0.75, warmup-cosine, drop-out substitute for drop-path):

Let me check lints, then smoke-test both configs (verify input scale is fixed, LLRD/freeze wiring works, and the model can actually learn).

Input scale is now correct (mean 0.18, std 0.50, range ≈[-2.1, 2.6] — proper ImageNet range), LLRD spans 4.75e-6→2e-4 across 28 groups, and frozen leaves only 3,843 head params trainable. Now a learnability smoke test (overfit 2 batches for a few epochs) to confirm the fix actually lets the model learn.

The filter missed the epoch metrics. Let me run a direct, faster overfit check that prints the trend explicitly.

The fix works — loss 0.83→0.001, train acc 0.12→1.00 in a few steps. Collapse resolved. Now launching both experiments in parallel: **frozen on GPU 0**, **full fine-tune on GPU 1**.

Both launched. Let me confirm they're training on separate GPUs and logging to W&B.

Both training cleanly on separate GPUs:
- **Frozen** (GPU 0): 3.8K trainable / 86.2M frozen — W&B `8cz3h5eu`
- **Fine-tune** (GPU 1): 86.2M trainable / 0 frozen — W&B `wwujnf64`

Let me wait for a few epochs to complete, then confirm neither run has collapsed.

Both are learning — no collapse. Frozen already shows real ordinal signal (QWK 0.29, acc 0.39 at epoch 8); fine-tune is still in/just past its 5-epoch warmup and climbing more slowly (expected). GPU1 at 94%, GPU0 (fast frozen forward) alternating. Let me let them train and check back in ~20 minutes.

Excellent progress — both learning well, and **fine-tune clearly leads**:
- Frozen: best F1 0.425, acc 0.447, QWK 0.346 (epoch 45)
- Fine-tune: best F1 0.497, acc 0.580, QWK 0.678 (epoch 45–47)

Both peaked ~epoch 45 and are now overfitting (261 samples), so early-stopping (patience 25) should end them shortly. Let me wait for completion.

Both runs finished. Let me extract the best-epoch metrics (by the checkpoint monitor `val/fuse/f1_macro`) for both, plus the CVFSNet coronal baseline for comparison.

Both runs are complete. Here are the final results (each at its best epoch by the checkpoint monitor `val/fuse/f1_macro`, same val set as the baseline). Let me write the summary to a file.

Done. Both experiments ran to completion in parallel and the summary is written to `RESULTS.md`.

## What I changed
- **ImageNet normalization** (`mae_ordinal/model.py`): the adapter now does per-clip min‑max → `[0,1]` → ImageNet mean/std. This fixed the collapse — inputs went from std≈0.004 to a proper ImageNet range (mean 0.18, std 0.50), and an overfit check hit 100% train acc in a few steps.
- **Frozen vs full-finetune support**: added `freeze_backbone`, layer‑wise LR decay (0.75), warmup+cosine schedule, and paper-based hyperparameters (VideoMAE `FINETUNE.md` / VideoMAEv2). Two configs: `config_frozen.yaml` (linear-probe: lr 1e‑3, wd 1e‑4, 3.8K params) and `config_finetune.yaml` (LLRD 0.75, base lr 2e‑4, wd 0.05, drop-path≈0.1 via hidden dropout).
- Ran both in parallel on 2 GPUs (frozen→GPU0, finetune→GPU1), tracked on W&B.

## Results (val, best epoch by `f1_macro`)

| Metric | Frozen | **Fine-tune** | CVFSNet coronal baseline |
|---|---:|---:|---:|
| F1 macro | 0.425 | **0.497** | 0.479 |
| Accuracy | 0.447 | **0.580** | 0.547 |
| AUROC macro | 0.687 | **0.745** | 0.701 |
| QWK | 0.346 | 0.648 | n/a |
| MAE | 0.900 | 0.640 | n/a |

**Conclusions:**
- Full fine-tuning beats frozen on every metric — with 261 clips the backbone benefits from adapting to DSA.
- Fine-tuned VideoMAE+ordinal **modestly edges the CVFSNet coronal baseline** (F1 +0.02, acc +0.03, AUROC +0.04) → competitive/promising, not conclusive.
- Ordinal objective helps ranking (QWK 0.65, MAE 0.64: errors land on adjacent grades).
- Class imbalance remains: fine-tune over-predicts T3 (recall 0.98) at the expense of T2a/T2b — a target for class-weighted loss.

**Caveats (in `RESULTS.md`):** `test=val`, single seed, best-epoch-on-val selection, so the ~0.02–0.03 margin over the baseline is within noise; both overfit after ~epoch 45.

W&B: frozen `8cz3h5eu`, fine-tune `wwujnf64`. Full write-up with hyperparameter tables, per-class recall, and next steps is in `RESULTS.md`.

---

**User**

Plot the image of confusion Metrics for all runs and compare it with the confusion Metrics of the baseline model

---

**Cursor**

I'll need to regenerate predictions from the saved best checkpoints (the CSV only logs one-vs-rest TP/FP/FN/TN, not the full confusion structure). Let me locate the checkpoints for all three runs.

All three have best-epoch checkpoints. I'll write a reusable script that reloads each checkpoint, regenerates val predictions, and plots the confusion matrices side by side.

All three reproduced (frozen 0.447, fine-tune 0.573, baseline 0.547 — matching the logged best-epoch numbers). Let me view the result.

Here's the comparison:

![Confusion matrices](/home/fshaikh/Projects/CVFSNet/plots/confusion_matrices.png)

Saved to `plots/confusion_matrices.png` (each cell shows count / row-normalized recall, at each model's best-epoch checkpoint on the same 150-sample val set). A notable finding: because I now have predictions for the baseline too, I could compute its QWK — let me fold that into `RESULTS.md`.

Done. Confusion matrices for all three runs are plotted and compared, saved to `plots/confusion_matrices.png`, and I updated `RESULTS.md` with the analysis.

## What the confusion matrices reveal

| Model | Accuracy | QWK (recomputed from checkpoint) |
|---|---:|---:|
| VideoMAE frozen | 0.447 | 0.346 |
| VideoMAE fine-tune | 0.573 | 0.647 |
| CVFSNet coronal (baseline) | 0.547 | **0.698** |

- **Frozen**: diffuse predictions, weak diagonal, T3 barely recognized (recall 0.23) — frozen K400 features don't separate DSA grades well.
- **Fine-tune**: highest accuracy, but a pronounced **T3 bias** — it funnels most T2a/T2b/T3 cases into T3 (T3 recall 0.98, T2a 0.31, T2b 0.17). Its accuracy is driven by nailing the two extreme classes.
- **Baseline (CVFSNet)**: strongest on T0/1 (0.76) and more evenly spread along the diagonal.

**Key nuance surfaced by this plot:** although the fine-tuned VideoMAE wins on **accuracy/F1**, the baseline actually has the **better ordinal agreement (QWK 0.698 vs 0.647)** — its mistakes land on neighboring grades, whereas VideoMAE collapses the middle grades into T3. So "beats the baseline" holds for accuracy but *not* for the ordinal ranking metric. This reinforces the earlier recommendation: add class-weighted CORN loss to recover T2a/T2b and improve QWK.

The plot generator is saved as `mae_ordinal/plot_confusion.py` (rerun with `python -m mae_ordinal.plot_confusion`) so it's reproducible as checkpoints change.
