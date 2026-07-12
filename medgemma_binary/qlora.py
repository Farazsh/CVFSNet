"""E2: QLoRA fine-tuning of MedGemma 1.5 4B for binary mTICI classification.

The frozen 4-bit base model is adapted with LoRA; a small classification head
reads the final hidden state at the last prompt token and is trained with BCE.

    python -m medgemma_binary.qlora --phase smoke
    python -m medgemma_binary.qlora --phase train --seed 14207
    python -m medgemma_binary.qlora --phase final --seed 14207

Which studies select the checkpoint and threshold depends on
``selection.protocol`` (see ``data.py``). Under ``internal`` they are the 53
tuning studies and the 150-study evaluation split is read only by
``--phase final``. Under ``full_train_val`` the run trains on all 261 train
studies and early-stops on those same 150 val studies, so ``--phase final``
re-scores the split the run already selected on -- a reporting convenience, not
an independent evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from .common import (
    REPO_ROOT,
    apply_overrides,
    environment_metadata,
    load_config,
    load_dotenv,
    scan_to_rgb_frames,
    seed_everything,
    summarize_rows,
    write_artifacts,
)

DEFAULT_CONFIG = Path(__file__).with_name("config_qlora_ap.yaml")
MONITORS = ("f1_macro_then_auroc", "auroc")


def resolve_monitor(training_cfg: Mapping[str, Any]) -> str:
    """The quantity early stopping maximizes.

    ``f1_macro_then_auroc`` (the default, and wave 1's behaviour) ranks epochs by
    macro F1 at that epoch's own tuned threshold, breaking ties on AUROC.
    ``auroc`` ranks on AUROC alone: any thresholded metric tracks the sigmoid's
    scale, which wave 1 showed is a seed artifact rather than a property of the
    model's discrimination.
    """
    monitor = str(training_cfg.get("monitor", "f1_macro_then_auroc"))
    if monitor not in MONITORS:
        raise ValueError(f"Unknown training.monitor '{monitor}'; expected one of {MONITORS}.")
    return monitor


def eval_split_label(config: Mapping[str, Any]) -> str:
    """What the early-stopping split should be *called* in the artifacts.

    Under ``selection.protocol: full_train_val`` that split is the val set, and a
    file named ``tuning_seed14207_scores.csv`` full of val scores is a trap for
    whoever reads the run directory next.
    """
    return str(config["run"].get("eval_split_label", "tuning"))


# --------------------------------------------------------------------------- #
# Prompt construction (frame totals come from the data, never from the template)
# --------------------------------------------------------------------------- #
def build_messages(
    prompt: Mapping[str, Any],
    views: Sequence[str],
    per_view_images: Sequence[Sequence[Image.Image]],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": str(prompt["preamble"])}]
    for view, images in zip(views, per_view_images):
        content.append({"type": "text", "text": f"{view.upper()} VIEW"})
        for frame_index, image in enumerate(images, start=1):
            content.append({"type": "text", "text": f"FRAME {frame_index} OF {len(images)}"})
            content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": str(prompt["question"])})
    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def resolve_target_modules(model: nn.Module, scope: str) -> list[str]:
    """Enumerate the fully-qualified Linear modules LoRA should adapt.

    ``all_linear_language`` reproduces the official MedGemma fine-tuning recipe's
    ``all-linear`` targeting while keeping the vision tower frozen, which is what
    the experiment plan asks for. The other scopes exist for ablations.

    Fully-qualified names matter: PEFT matches bare leaf names such as ``q_proj``
    by suffix, which silently also adapts the SigLIP vision encoder's attention
    projections. Returning full paths is what actually keeps the tower frozen.
    """
    attention = ("q_proj", "k_proj", "v_proj", "o_proj")
    names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name.endswith("lm_head"):
            continue
        leaf = name.rsplit(".", 1)[-1]
        in_language = ".language_model." in f".{name}"
        in_projector = "multi_modal_projector" in name
        if scope == "attention_only":
            keep = in_language and leaf in attention
        elif scope == "all_linear_language":
            keep = in_language
        elif scope == "all_linear_language_projector":
            keep = in_language or in_projector
        elif scope == "all_linear":
            keep = True
        else:
            raise ValueError(f"Unknown lora.target_scope '{scope}'.")
        if keep:
            names.append(name)
    if not names:
        raise ValueError(f"LoRA target scope '{scope}' matched no Linear modules.")
    return names


class MedGemmaBinaryClassifier(nn.Module):
    """4-bit MedGemma trunk + LoRA adapters + a BCE classification head."""

    def __init__(self, config: Mapping[str, Any], device: torch.device) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        model_cfg = config["model"]
        revision = str(model_cfg.get("revision") or "")
        if not revision:
            raise ValueError("model.revision must be pinned before training.")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        self.processor = AutoProcessor.from_pretrained(
            model_cfg["pretrained_name"],
            revision=revision,
            token=token,
            use_fast=bool(model_cfg.get("use_fast_processor", False)),
        )

        quant_cfg = config["quantization"]
        if not quant_cfg.get("enabled", True):
            raise ValueError("qlora.py implements the 4-bit strategy; set quantization.enabled.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=int(quant_cfg["bits"]) == 4,
            bnb_4bit_quant_type=str(quant_cfg["quant_type"]),
            bnb_4bit_use_double_quant=bool(quant_cfg["double_quant"]),
            bnb_4bit_compute_dtype=getattr(torch, str(quant_cfg["compute_dtype"])),
            bnb_4bit_quant_storage=getattr(torch, str(quant_cfg["compute_dtype"])),
        )
        base = AutoModelForImageTextToText.from_pretrained(
            model_cfg["pretrained_name"],
            revision=revision,
            token=token,
            dtype=getattr(torch, str(quant_cfg["compute_dtype"])),
            quantization_config=quantization_config,
            attn_implementation=str(model_cfg.get("attn_implementation", "sdpa")),
            device_map={"": device.index or 0},
            low_cpu_mem_usage=True,
        )
        base.config.use_cache = False

        training_cfg = config["training"]
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=bool(training_cfg["gradient_checkpointing"]),
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        lora_cfg = config["lora"]
        targets = resolve_target_modules(base, str(lora_cfg["target_scope"]))
        self.lora_target_modules = targets
        self.backbone = get_peft_model(
            base,
            LoraConfig(
                r=int(lora_cfg["rank"]),
                lora_alpha=int(lora_cfg["alpha"]),
                lora_dropout=float(lora_cfg["dropout"]),
                bias="none",
                target_modules=targets,
                task_type="FEATURE_EXTRACTION",
            ),
        )
        # ``get_peft_model`` swaps the Linear layers in place, so the multimodal
        # trunk reached through the base model already carries the adapters. We
        # call the trunk directly because the classification head replaces the
        # language-model head entirely.
        self.trunk = self.backbone.get_base_model().model

        unfrozen = [n for n, p in self.trunk.vision_tower.named_parameters() if p.requires_grad]
        if unfrozen:
            raise RuntimeError(
                f"The vision tower must stay frozen but {len(unfrozen)} parameters require "
                f"gradients (e.g. {unfrozen[0]}). Check lora.target_scope."
            )
        # With the tower frozen its activations never enter the backward graph,
        # so checkpointing it only pays for a recompute nothing consumes.
        if hasattr(self.trunk.vision_tower, "gradient_checkpointing_disable"):
            self.trunk.vision_tower.gradient_checkpointing_disable()

        hidden_size = int(base.config.text_config.hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(float(model_cfg["head_dropout"])),
            nn.Linear(hidden_size, 1),
        ).to(device=device, dtype=torch.float32)

        self.device = device
        self.compute_dtype = getattr(torch, str(quant_cfg["compute_dtype"]))
        self.expected_image_size = int(config["input"]["image_size"])
        self.expected_images = int(config["input"]["num_frames"]) * len(config["input"]["views"])

    def encode(self, messages: list[dict[str, Any]], study_name: str) -> dict[str, torch.Tensor]:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        pixels = inputs.get("pixel_values")
        expected = (self.expected_images, 3, self.expected_image_size, self.expected_image_size)
        if pixels is None or tuple(pixels.shape) != expected:
            shape = None if pixels is None else tuple(pixels.shape)
            raise ValueError(f"Processor pixels for {study_name} are {shape}, expected {expected}.")
        return inputs.to(self.device, dtype=self.compute_dtype)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.trunk(**inputs, use_cache=False)
        # Batch size is 1, so the final position is the last prompt token and no
        # padding mask arithmetic is required.
        last_hidden = outputs.last_hidden_state[:, -1, :]
        return self.head(last_hidden.float()).squeeze(-1)

    def trainable_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        adapters = [p for n, p in self.backbone.named_parameters() if p.requires_grad and "lora_" in n]
        head = list(self.head.parameters())
        if not adapters:
            raise RuntimeError("No LoRA parameters are trainable.")
        return adapters, head

    def parameter_counts(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable_parameters": trainable, "total_parameters": total}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _study_rows(
    model: MedGemmaBinaryClassifier,
    loader: DataLoader,
    config: Mapping[str, Any],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    prompt = config["prompts"][str(config["selection"]["prompt_id"])]
    scaling = config["intensity_scaling"][str(config["selection"]["scaling_id"])]
    views = list(config["input"]["views"])
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            name = batch["name"][0]
            images = [scan_to_rgb_frames(batch[view][0], scaling) for view in views]
            logit = model(model.encode(build_messages(prompt, views, images), name))
            value = float(logit.detach().float().view(-1)[0].item())
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite logit for {name}.")
            probability = float(torch.sigmoid(torch.tensor(value)).item())
            rows.append(
                {
                    "name": name,
                    "true_label": int(batch["label"].view(-1)[0].item()),
                    "logit": value,
                    "probability_t2b3": probability,
                    "predicted_label": int(probability >= threshold),
                    "exact_tie": False,
                    "threshold": float(threshold),
                    "experiment": str(config["run"]["name"]),
                    "seed": int(config["seed"]),
                    "num_frames": int(config["input"]["num_frames"]),
                    "views": "+".join(views),
                }
            )
    return rows


def select_threshold(rows: Sequence[Mapping[str, Any]]) -> float:
    """Pick the tuning-split threshold that maximizes macro F1."""
    from sklearn.metrics import f1_score

    y_true = np.asarray([int(row["true_label"]) for row in rows])
    y_score = np.asarray([float(row["probability_t2b3"]) for row in rows])
    candidates = np.unique(np.concatenate([[0.5], y_score]))
    best_threshold, best_score = 0.5, -1.0
    for threshold in candidates:
        score = f1_score(y_true, (y_score >= threshold).astype(int), average="macro", zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def _rescore(rows: Sequence[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "threshold": float(threshold),
            "predicted_label": int(float(row["probability_t2b3"]) >= threshold),
        }
        for row in rows
    ]


def _summaries(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    tuned_threshold: float,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    replicates = int(evaluation["bootstrap_replicates"])
    seed = int(config["seed"])
    return {
        "at_0.5": summarize_rows(
            _rescore(rows, 0.5), threshold=0.5, bootstrap_replicates=replicates, seed=seed
        ),
        "at_tuned_threshold": summarize_rows(
            _rescore(rows, tuned_threshold),
            threshold=tuned_threshold,
            bootstrap_replicates=replicates,
            seed=seed,
        ),
        "tuned_threshold": float(tuned_threshold),
    }


def _calibration(rows: Sequence[Mapping[str, Any]], bins: int = 10) -> dict[str, float]:
    y_true = np.asarray([int(row["true_label"]) for row in rows], dtype=float)
    y_prob = np.asarray([float(row["probability_t2b3"]) for row in rows])
    brier = float(np.mean((y_prob - y_true) ** 2))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (y_prob > low) & (y_prob <= high) if low > 0 else (y_prob >= low) & (y_prob <= high)
        if mask.any():
            ece += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return {"brier": brier, "expected_calibration_error": float(ece)}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def run_dir(config: Mapping[str, Any]) -> Path:
    path = REPO_ROOT / str(config["run"]["output_dir"]) / str(config["run"]["name"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def train(config: Mapping[str, Any], data, model: MedGemmaBinaryClassifier, logger) -> dict[str, Any]:
    from transformers import get_cosine_schedule_with_warmup

    training_cfg = config["training"]
    optimizer_cfg = config["optimizer"]
    prompt = config["prompts"][str(config["selection"]["prompt_id"])]
    scaling = config["intensity_scaling"][str(config["selection"]["scaling_id"])]
    views = list(config["input"]["views"])
    monitor = resolve_monitor(training_cfg)
    label = eval_split_label(config)

    adapters, head = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        [
            {"params": adapters, "lr": float(optimizer_cfg["adapter_lr"])},
            {"params": head, "lr": float(optimizer_cfg["head_lr"])},
        ],
        weight_decay=float(optimizer_cfg["weight_decay"]),
        fused=True,
    )

    accumulation = int(training_cfg["gradient_accumulation_steps"])
    max_epochs = int(training_cfg["max_epochs"])
    train_loader = data.loader("development", shuffle=True)
    tuning_loader = data.loader("tuning", shuffle=False)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = steps_per_epoch * max_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(round(float(training_cfg["warmup_ratio"]) * total_steps)),
        num_training_steps=total_steps,
    )

    criterion = nn.BCEWithLogitsLoss()
    clip_value = float(training_cfg["gradient_clip_val"])
    patience = int(training_cfg["early_stopping_patience"])
    checkpoint_dir = run_dir(config) / f"seed_{int(config['seed'])}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats(model.device)
    best = {"macro_f1": -1.0, "auroc": -1.0, "epoch": -1, "threshold": 0.5}
    history: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        model.trunk.train()
        epoch_started = time.perf_counter()
        running_loss, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for index, batch in enumerate(train_loader):
            name = batch["name"][0]
            images = [scan_to_rgb_frames(batch[view][0], scaling) for view in views]
            logit = model(model.encode(build_messages(prompt, views, images), name))
            target = batch["label"].view(-1).to(model.device, dtype=torch.float32)
            loss = criterion(logit.view(1), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss on {name}.")
            (loss / accumulation).backward()

            running_loss += float(loss.detach().item())
            seen += 1
            is_last = index == len(train_loader) - 1
            if (index + 1) % accumulation == 0 or is_last:
                torch.nn.utils.clip_grad_norm_(adapters + head, clip_value)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if logger is not None and global_step % int(training_cfg["log_every_steps"]) == 0:
                    logger.log(
                        {
                            "train/loss": running_loss / max(seen, 1),
                            "train/adapter_lr": scheduler.get_last_lr()[0],
                            "train/head_lr": scheduler.get_last_lr()[1],
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

        train_loss = running_loss / max(seen, 1)
        tuning_rows = _study_rows(model, tuning_loader, config, threshold=0.5)
        tuning_summary = summarize_rows(
            _rescore(tuning_rows, 0.5),
            threshold=0.5,
            bootstrap_replicates=0,
            seed=int(config["seed"]),
        )
        metrics = tuning_summary["metrics"]
        # Select the epoch under the same decision rule the final split will use:
        # macro F1 at the threshold this epoch's tuning scores would choose. A
        # head that has not yet calibrated collapses to one class at 0.5, which
        # would otherwise make every early epoch look identically bad.
        epoch_threshold = select_threshold(tuning_rows)
        epoch_f1 = summarize_rows(
            _rescore(tuning_rows, epoch_threshold),
            threshold=epoch_threshold,
            bootstrap_replicates=0,
            seed=int(config["seed"]),
        )["metrics"]["f1_macro"]
        tuning_loss = float(
            nn.functional.binary_cross_entropy(
                torch.tensor([r["probability_t2b3"] for r in tuning_rows], dtype=torch.float64).clamp(1e-7, 1 - 1e-7),
                torch.tensor([float(r["true_label"]) for r in tuning_rows], dtype=torch.float64),
            ).item()
        )
        record = {
            "epoch": epoch,
            "train/loss": train_loss,
            f"{label}/loss": tuning_loss,
            f"{label}/loss_gap": tuning_loss - train_loss,
            f"{label}/f1_macro_at_0.5": metrics["f1_macro"],
            f"{label}/f1_macro_at_tuned": epoch_f1,
            f"{label}/tuned_threshold": epoch_threshold,
            f"{label}/auroc": metrics["auroc"],
            f"{label}/balanced_accuracy": metrics["balanced_accuracy"],
            f"{label}/accuracy": metrics["accuracy"],
            "epoch_seconds": time.perf_counter() - epoch_started,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(model.device)),
        }
        record.update({f"{label}/{k}": v for k, v in _calibration(tuning_rows).items()})
        history.append(record)
        if logger is not None:
            logger.log(record, step=global_step)
        print(
            f"epoch {epoch:3d} | train loss {train_loss:.4f} | {label} loss {tuning_loss:.4f} "
            f"| {label} macro-F1 {epoch_f1:.4f} @ thr {epoch_threshold:.3f} "
            f"(F1@0.5 {metrics['f1_macro']:.4f}) | AUROC {metrics['auroc']:.4f}",
            flush=True,
        )

        if monitor == "auroc":
            improved = metrics["auroc"] > best["auroc"]
        else:
            improved = (epoch_f1, metrics["auroc"]) > (best["macro_f1"], best["auroc"])
        if improved:
            best = {
                "macro_f1": epoch_f1,
                "auroc": metrics["auroc"],
                "epoch": epoch,
                "threshold": epoch_threshold,
            }
            epochs_without_improvement = 0
            model.backbone.save_pretrained(str(checkpoint_dir / "adapter"))
            torch.save(model.head.state_dict(), checkpoint_dir / "head.pt")
            best_rows = tuning_rows
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}.", flush=True)
                break

    tuned_threshold = best["threshold"]
    selection = {
        "best_epoch": best["epoch"],
        "tuning_macro_f1_at_tuned_threshold": best["macro_f1"],
        "tuning_auroc": best["auroc"],
        "tuned_threshold": tuned_threshold,
        "selection_metric": monitor,
        "eval_split_label": label,
        "selection_protocol": str(config["selection"].get("protocol", "internal")),
        "wall_seconds": time.perf_counter() - started,
        "seconds_per_training_study": (time.perf_counter() - started)
        / max(len(train_loader) * len(history), 1),
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(model.device)),
        "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        **model.parameter_counts(),
        "lora_target_modules": model.lora_target_modules,
    }
    with open(checkpoint_dir / "selection.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(selection, handle, sort_keys=False)
    with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    tuning_summaries = _summaries(best_rows, config, tuned_threshold)
    write_artifacts(
        config=config,
        split_name=f"{label}_seed{int(config['seed'])}",
        rows=_rescore(best_rows, tuned_threshold),
        summary={**tuning_summaries, **selection},
        metadata=environment_metadata(str(config["model"]["revision"])),
    )
    if logger is not None:
        logger.log(
            {
                f"best/{label}_f1_macro_at_tuned_threshold": best["macro_f1"],
                f"best/{label}_auroc": best["auroc"],
                "best/epoch": best["epoch"],
                "best/tuned_threshold": tuned_threshold,
            },
            step=global_step,
        )
    return selection


def evaluate_final(config: Mapping[str, Any], data, model: MedGemmaBinaryClassifier, logger) -> dict[str, Any]:
    checkpoint_dir = run_dir(config) / f"seed_{int(config['seed'])}"
    selection_path = checkpoint_dir / "selection.yaml"
    if not selection_path.exists():
        raise FileNotFoundError(f"{selection_path} is missing; run --phase train first.")
    with open(selection_path, "r", encoding="utf-8") as handle:
        selection = yaml.safe_load(handle)

    # The model already owns a "default" adapter from get_peft_model, so load the
    # trained weights into it rather than registering a second adapter.
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    state = load_file(str(checkpoint_dir / "adapter" / "adapter_model.safetensors"))
    result = set_peft_model_state_dict(model.backbone, state)
    if getattr(result, "unexpected_keys", []):
        raise RuntimeError(f"Unexpected adapter keys: {result.unexpected_keys[:3]}")
    model.head.load_state_dict(torch.load(checkpoint_dir / "head.pt", map_location=model.device))

    threshold = float(selection["tuned_threshold"])
    rows = _study_rows(model, data.loader("final", shuffle=False), config, threshold=threshold)
    summaries = _summaries(rows, config, threshold)
    summaries.update(_calibration(rows))
    summaries["selected_epoch"] = selection["best_epoch"]
    write_artifacts(
        config=config,
        split_name=f"final_seed{int(config['seed'])}",
        rows=rows,
        summary=summaries,
        metadata=environment_metadata(str(config["model"]["revision"])),
    )
    if logger is not None:
        logger.log(
            {
                f"final/{key}": value
                for key, value in {
                    **{f"at_0.5/{k}": v for k, v in summaries["at_0.5"]["metrics"].items()},
                    **{
                        f"at_tuned/{k}": v
                        for k, v in summaries["at_tuned_threshold"]["metrics"].items()
                    },
                }.items()
            }
        )
    return summaries


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def validate_runtime() -> None:
    import transformers
    from packaging.version import Version

    if Version(torch.__version__.split("+")[0]) < Version("2.6.0"):
        raise RuntimeError("MedGemma requires the isolated torch>=2.6 environment.")
    if Version(transformers.__version__) < Version("4.57.1"):
        raise RuntimeError("MedGemma 1.5 requires transformers>=4.57.1.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("QLoRA training requires a BF16-capable CUDA GPU.")


def make_logger(config: Mapping[str, Any], phase: str):
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    import wandb

    return wandb.init(
        project=str(wandb_cfg["project"]),
        entity=wandb_cfg.get("entity"),
        group=str(wandb_cfg["group"]),
        name=f"{config['run']['name']}_seed{int(config['seed'])}_{phase}",
        job_type=phase,
        config=json.loads(json.dumps(dict(config))),
        reinit=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("smoke", "train", "final"), default="smoke")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke only: shrink the development and tuning sets to this many studies.",
    )
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def balanced_subset(names: Sequence[str], labels: Sequence[int], count: int) -> list[str]:
    by_class: dict[int, list[str]] = {}
    for name, label in zip(names, labels):
        by_class.setdefault(int(label), []).append(name)
    per_class = max(count // len(by_class), 1)
    return [name for label in sorted(by_class) for name in by_class[label][:per_class]]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    # Smoke defaults go in first so that an explicit --set overrides them rather
    # than being silently discarded by them.
    if args.phase == "smoke":
        config["training"]["max_epochs"] = 1
        config["training"]["early_stopping_patience"] = 1
        config["evaluation"]["bootstrap_replicates"] = 100
    apply_overrides(config, args.set)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return

    validate_runtime()
    load_dotenv()
    seed_everything(int(config["seed"]))

    from .data import MedGemmaData

    data = MedGemmaData(config)
    label = eval_split_label(config)
    images_per_study = int(config["input"]["num_frames"]) * len(config["input"]["views"])
    # Print the real split before --limit truncates it: the sizes are what the
    # smoke test is checking.
    print(
        f"protocol={data.protocol} monitor={resolve_monitor(config['training'])} "
        f"eval_split_label={label}\n"
        f"development={len(data.development)} tuning={len(data.tuning)} final={len(data.final)} "
        f"views={config['input']['views']} frames={config['input']['num_frames']} "
        f"images/study={images_per_study}",
        flush=True,
    )
    if args.limit:
        if args.phase != "smoke":
            raise ValueError("--limit is a smoke-test convenience and must not shrink a real run.")
        data.development = balanced_subset(
            data.development, data.labels_for(data.development), args.limit
        )
        data.tuning = balanced_subset(data.tuning, data.labels_for(data.tuning), args.limit)
        print(
            f"--limit {args.limit}: development={len(data.development)} "
            f"tuning={len(data.tuning)}",
            flush=True,
        )

    logger = make_logger(config, args.phase)
    device = torch.device("cuda", 0)
    model = MedGemmaBinaryClassifier(config, device)
    counts = model.parameter_counts()
    targets = model.lora_target_modules
    print(
        f"LoRA targets: {len(targets)} modules "
        f"(e.g. {targets[0]} ... {targets[-1]})\n"
        f"trainable={counts['trainable_parameters']:,} / total={counts['total_parameters']:,}",
        flush=True,
    )

    try:
        if args.phase == "final":
            summaries = evaluate_final(config, data, model, logger)
        else:
            summaries = train(config, data, model, logger)
        print(yaml.safe_dump(json.loads(json.dumps(summaries)), sort_keys=False))
    finally:
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    main()
