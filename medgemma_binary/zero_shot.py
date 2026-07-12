"""Zero-shot MedGemma binary TICI classification.

This runner reuses the repository's existing trilinear resampling pipeline and
scores fixed answers ("0" vs "1") instead of doing free-form generation.

Example:
    CUDA_VISIBLE_DEVICES=0 uv run python -m medgemma_binary.zero_shot \
        --config medgemma_binary/config_zero_shot.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForImageTextToText, AutoProcessor
try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None  # type: ignore[assignment]

from amticis_pipeline.dataset_loader import AmTICISDataModule
from amticis_training.classification_metrics import ClassificationMetricAccumulator
from amticis_training.config import build_pipeline_config, deep_update, parse_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config_zero_shot.yaml")
OUTPUT_NAME = "fuse"
CLASS_NAMES = ["T012a", "T2b3"]


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except Exception:
        pass
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--split",
        default=None,
        choices=("train", "val", "test"),
        help="Override the evaluation split from the YAML config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use. Default: cuda if available, otherwise cpu.",
    )
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def load_config(path: Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    deep_update(config, parse_overrides(overrides))
    return config


def _torch_dtype(name: str | None, device: str) -> torch.dtype:
    if device.startswith("cpu"):
        return torch.float32
    mapping = {
        None: torch.float16,
        "auto": torch.float16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'.")
    return mapping[name]


def _maybe_bitsandbytes_config(model_cfg: Mapping[str, Any]) -> Any:
    quantization = str(model_cfg.get("quantization", "")).lower()
    if not quantization or BitsAndBytesConfig is None:
        return None
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, str(model_cfg.get("bnb_4bit_compute_dtype", "float16"))),
            bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization mode '{quantization}'.")


def _ensure_set_submodule_compat() -> None:
    if hasattr(torch.nn.Module, "set_submodule"):
        return

    def _set_submodule(self: torch.nn.Module, target: str, module: torch.nn.Module) -> None:
        parts = target.split(".")
        parent: torch.nn.Module = self
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)

    torch.nn.Module.set_submodule = _set_submodule  # type: ignore[attr-defined]


def _model_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _patch_transformers_masking() -> None:
    """Allow Gemma3's masking helper to run on the repo's torch version.

    The current transformers build gates ``or_mask_function`` / ``and_mask_function``
    behind a torch>=2.6 check even though the underlying logic still executes on
    this environment. We flip the runtime flag before model loading so the zero-shot
    pass can run without upgrading the entire repo stack.
    """
    try:
        from contextlib import nullcontext
        import transformers.models.gemma3.modeling_gemma3 as gemma3
        import transformers.masking_utils as masking_utils

        masking_utils._is_torch_greater_or_equal_than_2_6 = True
        if not hasattr(masking_utils, "TransformGetItemToIndex"):
            masking_utils.TransformGetItemToIndex = nullcontext  # type: ignore[attr-defined]

        def _simple_create_masks_for_vision_model(
            config,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids,
            block_sequence_ids,
        ):
            mask_kwargs = {
                "config": config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "block_sequence_ids": block_sequence_ids,
            }
            full_mask = masking_utils.create_causal_mask(**mask_kwargs)
            return {"full_attention": full_mask, "sliding_attention": full_mask}

        gemma3.create_masks_for_vision_model = _simple_create_masks_for_vision_model  # type: ignore[assignment]
    except Exception:
        pass


def _move_to_device(batch: Mapping[str, torch.Tensor], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _tensor_device_from_map(device_map: Mapping[str, Any], keys: Sequence[str], default: str) -> str:
    for key in keys:
        if key in device_map:
            value = device_map[key]
            if isinstance(value, int):
                return f"cuda:{value}"
            return str(value)
    return default


def _route_processor_inputs(
    inputs: Mapping[str, Any],
    *,
    text_device: str,
    vision_device: str,
) -> dict[str, Any]:
    routed: dict[str, Any] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            routed[key] = value
            continue

        if key.startswith("pixel_") or "image" in key or "vision" in key or key.endswith("grid_thw"):
            routed[key] = value.to(vision_device, non_blocking=True)
        else:
            routed[key] = value.to(text_device, non_blocking=True)
    return routed


def _build_messages(
    views: Sequence[str],
    images: Sequence[Sequence[Image.Image]],
    num_frames: int,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Classify the final mTICI reperfusion score from these ordered DSA frames.\n"
                "Class 0 = T0, T1, T2A.\n"
                "Class 1 = T2B, T3.\n"
                "Reply with only 0 or 1.\n\n"
            ),
        }
    ]
    for view, view_images in zip(views, images):
        content.append({"type": "text", "text": f"{view.upper()} VIEW\n"})
        for frame_idx, frame in enumerate(view_images):
            content.append({"type": "text", "text": f"FRAME {frame_idx + 1} OF {num_frames}\n"})
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": "\n"})
    return [{"role": "user", "content": content}]


def _scan_to_rgb_frames(
    clip: torch.Tensor,
    scaling: Mapping[str, Any],
) -> list[Image.Image]:
    """Convert one resampled scan ``(1, T, H, W)`` into a list of RGB frames."""
    if clip.ndim != 4 or clip.shape[0] != 1:
        raise ValueError(f"Expected scan shape (1, T, H, W), got {tuple(clip.shape)}.")

    arr = clip.detach().cpu().float().numpy()[0]  # (T, H, W)
    mode = str(scaling.get("mode", "percentile")).lower()
    if mode == "percentile":
        lower = float(scaling.get("lower", 1.0))
        upper = float(scaling.get("upper", 99.0))
        lo = np.percentile(arr, lower)
        hi = np.percentile(arr, upper)
    elif mode == "minmax":
        lo = float(arr.min())
        hi = float(arr.max())
    else:
        raise ValueError(f"Unknown intensity scaling mode '{mode}'.")

    scaled = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    frames: list[Image.Image] = []
    for frame in scaled:
        gray = np.asarray(np.round(frame * 255.0), dtype=np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        frames.append(Image.fromarray(rgb, mode="RGB"))
    return frames


def _sample_inputs(
    batch: Mapping[str, Any],
    index: int,
    views: Sequence[str],
    scaling: Mapping[str, Any],
) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for view in views:
        frames.extend(_scan_to_rgb_frames(batch[view][index], scaling=scaling))
    return frames


def _encode_sample(
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    images: Sequence[Sequence[Image.Image]],
    messages: list[dict[str, Any]],
    answers: Sequence[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    device_map = getattr(model, "hf_device_map", {}) or {}
    text_device = _tensor_device_from_map(
        device_map,
        (
            "model.language_model.embed_tokens",
            "model.language_model",
            "lm_head",
        ),
        device,
    )
    vision_device = _tensor_device_from_map(
        device_map,
        (
            "model.vision_tower",
            "vision_tower",
        ),
        text_device,
    )

    base_inputs = processor.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    base_inputs = _route_processor_inputs(
        base_inputs,
        text_device=text_device,
        vision_device=vision_device,
    )

    if "input_ids" not in base_inputs:
        raise KeyError("Processor output is missing input_ids.")
    if "attention_mask" not in base_inputs:
        base_inputs["attention_mask"] = torch.ones_like(base_inputs["input_ids"])

    prompt_len = int(base_inputs["input_ids"].shape[1])
    candidate_scores: list[torch.Tensor] = []
    for answer in answers:
        answer_ids = processor.tokenizer.encode(answer, add_special_tokens=False)
        if not answer_ids:
            raise ValueError(f"Answer '{answer}' tokenized to an empty sequence.")

        full_inputs = dict(base_inputs)
        answer_tensor = torch.tensor([answer_ids], device=text_device, dtype=base_inputs["input_ids"].dtype)
        answer_mask = torch.ones_like(answer_tensor)
        full_inputs["input_ids"] = torch.cat([base_inputs["input_ids"], answer_tensor], dim=1)
        full_inputs["attention_mask"] = torch.cat([base_inputs["attention_mask"], answer_mask], dim=1)

        outputs = model(**full_inputs, use_cache=False)
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        target_ids = full_inputs["input_ids"][:, 1:]
        token_log_probs = log_probs[:, :-1].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        candidate_scores.append(token_log_probs[:, prompt_len - 1 :].sum(dim=1))

    scores = torch.cat(candidate_scores, dim=0)
    scores = torch.nan_to_num(scores, nan=-1e9, neginf=-1e9, posinf=1e9)
    return scores, base_inputs["input_ids"]


@torch.inference_mode()
def evaluate_split(config: Mapping[str, Any], split: str | None = None, device: str | None = None) -> dict[str, Any]:
    _load_dotenv()
    _patch_transformers_masking()
    _ensure_set_submodule_compat()
    runtime_device = _model_device(device)
    dtype = _torch_dtype(config.get("model", {}).get("dtype"), runtime_device)
    seed = int(config.get("seed", 1))
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_cfg = config["data"]
    pipeline_config = build_pipeline_config({"data": data_cfg})
    datamodule = AmTICISDataModule(config=pipeline_config)
    datamodule.setup("fit")

    evaluation_cfg = config.get("evaluation", {})
    split_name = split or evaluation_cfg.get("split", "val")
    loader = getattr(datamodule, f"{split_name}_dataloader")()
    views = list(datamodule.config.views.active)
    num_frames = int(datamodule.config.data.num_frames)
    scaling = evaluation_cfg.get("intensity_scaling", {})
    answers = list(evaluation_cfg.get("answer_labels", ["0", "1"]))

    model_cfg = config["model"]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    device_map = model_cfg.get("device_map")
    quantization_config = _maybe_bitsandbytes_config(model_cfg)
    model_kwargs = {
        "revision": model_cfg.get("revision"),
        "token": token,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if device_map:
        model_kwargs["device_map"] = device_map
    if model_cfg.get("offload_folder"):
        model_kwargs["offload_folder"] = str(REPO_ROOT / str(model_cfg["offload_folder"]))
    processor_kwargs = {
        "revision": model_cfg.get("revision"),
        "token": token,
    }
    processor = AutoProcessor.from_pretrained(model_cfg["pretrained_name"], **processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(model_cfg["pretrained_name"], **model_kwargs)
    if not device_map:
        model = model.to(runtime_device)
    model = model.eval()
    input_device = str(next(model.parameters()).device)

    metric_cfg = config.get("metrics", {})
    accumulator = ClassificationMetricAccumulator(
        num_classes=2,
        undefined_value=float(metric_cfg.get("undefined_value", 0.0)),
        log_per_class=bool(metric_cfg.get("log_per_class", True)),
        outputs=metric_cfg.get("outputs"),
    )

    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = _move_to_device(batch, runtime_device)
        labels = batch["label"].view(-1).long().cpu()
        names = batch["name"]
        if isinstance(names, str):
            names = [names]

        batch_scores: list[torch.Tensor] = []
        for sample_idx, name in enumerate(names):
            per_view_images = [
                _scan_to_rgb_frames(batch[view][sample_idx], scaling=scaling)
                for view in views
            ]
            sample_scores, _ = _encode_sample(
                processor=processor,
                model=model,
                images=per_view_images,
                messages=_build_messages(views, per_view_images, num_frames),
                answers=answers,
                device=input_device,
            )
            sample_scores = torch.nan_to_num(sample_scores.float(), nan=-1e9, neginf=-1e9, posinf=1e9)
            batch_scores.append(sample_scores.unsqueeze(0))

            probs = torch.softmax(sample_scores, dim=0)
            pred = int(torch.argmax(sample_scores).item())
            rows.append(
                {
                    "name": name,
                    "true_label": int(labels[sample_idx].item()),
                    "score_0": float(sample_scores[0].item()),
                    "score_1": float(sample_scores[1].item()),
                    "prob_1": float(probs[1].item()),
                    "pred_label": pred,
                }
            )

        score_tensor = torch.cat(batch_scores, dim=0).cpu()
        accumulator.update(OUTPUT_NAME, score_tensor, labels)

    state = accumulator.state()
    metrics = accumulator.compute_from_state(state)
    y_true = [row["true_label"] for row in rows]
    y_pred = [row["pred_label"] for row in rows]
    y_score = [float(np.nan_to_num(row["prob_1"], nan=0.5, posinf=1.0, neginf=0.0)) for row in rows]

    summary = {
        "run": str(config.get("run", {}).get("name", "medgemma_zero_shot_binary_tici")),
        "split": split_name,
        "n": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "auroc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.0,
        "metrics": metrics,
        "rows": rows,
    }

    _write_outputs(config, summary)
    return summary


def _write_outputs(config: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    run_cfg = config.get("run", {})
    run_dir = REPO_ROOT / str(run_cfg.get("output_dir", "output_runs")) / str(run_cfg.get("name", "medgemma_zero_shot_binary_tici"))
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_path = run_dir / "resolved_zero_shot_config.yaml"
    with open(resolved_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)

    csv_path = run_dir / "val_zero_shot_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "true_label", "score_0", "score_1", "prob_1", "pred_label"],
        )
        writer.writeheader()
        writer.writerows(summary["rows"])

    metrics_path = run_dir / "val_zero_shot_metrics.yaml"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "run": summary["run"],
                "split": summary["split"],
                "n": summary["n"],
                "accuracy": summary["accuracy"],
                "f1_macro": summary["f1_macro"],
                "auroc": summary["auroc"],
                "metrics": summary["metrics"],
            },
            handle,
            sort_keys=False,
        )

    cm_path = run_dir / "val_confusion_matrix.png"
    _write_confusion_matrix(cm_path, summary["rows"])
    _maybe_log_wandb(config, summary, csv_path, metrics_path, cm_path)

    print(
        f"{summary['run']}[{summary['split']}]: n={summary['n']} "
        f"acc={summary['accuracy']:.3f} f1={summary['f1_macro']:.3f} "
        f"auroc={summary['auroc']:.3f} -> {csv_path}"
    )


def _write_confusion_matrix(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    y_true = np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray([int(row["pred_label"]) for row in rows], dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.clip(row_sum, 1, None)

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("MedGemma zero-shot binary TICI confusion matrix")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _maybe_log_wandb(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    csv_path: Path,
    metrics_path: Path,
    cm_path: Path,
) -> None:
    logging_cfg = config.get("logging", {})
    wandb_cfg = logging_cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return
    api_key = os.environ.get(wandb_cfg.get("api_key_env", "WANDB_API_KEY"))
    if api_key:
        os.environ.setdefault("WANDB_API_KEY", api_key)
    if wandb_cfg.get("offline", False):
        os.environ["WANDB_MODE"] = "offline"
    try:
        import wandb
    except Exception:
        return

    run = wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        name=str(config.get("run", {}).get("name", "medgemma_zero_shot_binary_tici")),
        tags=wandb_cfg.get("tags"),
        config=dict(config),
        dir=str(csv_path.parent),
        reinit=True,
    )
    wandb.log(
        {
            "zero_shot/accuracy": summary["accuracy"],
            "zero_shot/f1_macro": summary["f1_macro"],
            "zero_shot/auroc": summary["auroc"],
            "zero_shot/n": summary["n"],
            "zero_shot/csv_path": str(csv_path),
            "zero_shot/metrics_path": str(metrics_path),
            "zero_shot/confusion_matrix": wandb.Image(str(cm_path)),
        }
    )
    wandb.finish()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    if args.split is not None:
        config.setdefault("evaluation", {})["split"] = args.split
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    evaluate_split(config=config, split=args.split, device=args.device)


if __name__ == "__main__":
    main()
