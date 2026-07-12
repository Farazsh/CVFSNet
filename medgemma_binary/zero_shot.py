"""Corrected MedGemma zero-shot binary TICI evaluation.

Run tuning first, inspect the generated selection, then run the locked final phase:

    python -m medgemma_binary.zero_shot --phase tuning
    python -m medgemma_binary.zero_shot --phase final
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml
from packaging.version import Version
from PIL import Image
from torch.utils.data import DataLoader, Subset

from amticis_pipeline.dataset_loader import AmTICISDataModule
from amticis_training.config import build_pipeline_config

from .common import (
    REPO_ROOT,
    environment_metadata,
    load_config,
    load_dotenv,
    make_internal_split,
    scan_to_rgb_frames,
    score_row,
    seed_everything,
    summarize_rows,
    write_artifacts,
    write_split_manifest,
)


DEFAULT_CONFIG = Path(__file__).with_name("config_zero_shot.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("smoke", "tuning", "final"), default="smoke")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def validate_runtime() -> None:
    """Reject the unsupported stack that required the previous mask patch."""
    import transformers

    if Version(torch.__version__.split("+")[0]) < Version("2.6.0"):
        raise RuntimeError(
            "MedGemma requires the isolated torch>=2.6 environment. "
            "Do not restore the previous attention-mask monkey patch."
        )
    if Version(transformers.__version__) < Version("4.57.1"):
        raise RuntimeError("MedGemma 1.5 requires transformers>=4.57.1.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The corrected primary run requires a BF16-capable CUDA GPU.")


def build_messages(
    prompt: Mapping[str, Any],
    views: Sequence[str],
    per_view_images: Sequence[Sequence[Image.Image]],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": str(prompt["preamble"])}]
    for view, images in zip(views, per_view_images):
        content.append({"type": "text", "text": f"{view.upper()} VIEW"})
        for frame_index, image in enumerate(images, start=1):
            content.append(
                {"type": "text", "text": f"FRAME {frame_index} OF {len(images)}"}
            )
            content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": str(prompt["question"])})
    return [{"role": "user", "content": content}]


def answer_token_ids(processor: Any, answers: Sequence[str]) -> list[int]:
    token_ids: list[int] = []
    for answer in answers:
        encoded = processor.tokenizer.encode(answer, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Answer '{answer}' must be one token, got {encoded}.")
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(f"Answer labels do not have distinct token IDs: {token_ids}.")
    return token_ids


def next_token_class_scores(
    logits: torch.Tensor,
    answer_ids: Sequence[int],
) -> torch.Tensor:
    """Select fixed-answer logits from a one-token model output."""
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != 1:
        raise ValueError(f"Expected logits shape (1, 1, vocab), got {tuple(logits.shape)}.")
    return logits[0, 0, torch.as_tensor(answer_ids, device=logits.device)]


def selected_lm_head_scores(
    hidden_state: torch.Tensor,
    lm_head_weight: torch.Tensor,
    answer_ids: Sequence[int],
) -> torch.Tensor:
    """Project only the two answer tokens with FP32 accumulation.

    The released model runs in BF16, but a two-way decision should not become
    an exact tie merely because the full vocabulary projection was rounded to
    BF16. The underlying hidden state and weights remain those of the pinned
    model; only the final dot products accumulate in FP32.
    """
    if hidden_state.ndim != 2 or hidden_state.shape[0] != 1:
        raise ValueError(
            f"Expected final hidden state shape (1, hidden), got {tuple(hidden_state.shape)}."
        )
    token_ids = torch.as_tensor(answer_ids, device=lm_head_weight.device)
    selected_weight = lm_head_weight.index_select(0, token_ids)
    selected_weight = selected_weight.to(device=hidden_state.device, dtype=torch.float32)
    return F.linear(hidden_state.float(), selected_weight).squeeze(0)


class MedGemmaScorer:
    def __init__(self, config: Mapping[str, Any]) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_cfg = config["model"]
        revision = str(model_cfg.get("revision") or "")
        if not revision:
            raise ValueError("model.revision must be pinned before evaluation.")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        common_kwargs = {
            "revision": revision,
            "token": token,
        }
        self.processor = AutoProcessor.from_pretrained(
            model_cfg["pretrained_name"],
            **common_kwargs,
            use_fast=bool(model_cfg.get("use_fast_processor", False)),
        )
        model_kwargs: dict[str, Any] = {
            **common_kwargs,
            "dtype": torch.bfloat16,
            "device_map": model_cfg.get("device_map", "auto"),
            "low_cpu_mem_usage": True,
            "attn_implementation": model_cfg.get("attn_implementation", "sdpa"),
        }
        if model_cfg.get("max_memory"):
            model_kwargs["max_memory"] = model_cfg["max_memory"]
        if model_cfg.get("offload_folder"):
            offload_folder = REPO_ROOT / str(model_cfg["offload_folder"])
            offload_folder.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(offload_folder)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_cfg["pretrained_name"], **model_kwargs
        ).eval()
        self.device = self.model.device
        self.answer_ids = answer_token_ids(
            self.processor, config["evaluation"]["answer_labels"]
        )
        self.expected_image_size = int(config["input"]["image_size"])
        self.expected_images = int(config["input"]["num_frames"]) * len(
            config["input"]["views"]
        )

    @torch.inference_mode()
    def score(
        self,
        *,
        messages: list[dict[str, Any]],
        study_name: str,
        expected_images: int | None = None,
    ) -> torch.Tensor:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        self._validate_processor_output(
            inputs,
            study_name,
            self.expected_images if expected_images is None else expected_images,
        )
        inputs = inputs.to(self.device, dtype=torch.bfloat16)
        outputs = self.model.model(**inputs, use_cache=False)
        final_hidden_state = outputs.last_hidden_state[:, -1, :]
        scores = selected_lm_head_scores(
            final_hidden_state,
            self.model.lm_head.weight,
            self.answer_ids,
        )
        if not torch.isfinite(scores).all():
            raise FloatingPointError(
                f"Non-finite MedGemma logits for {study_name}: "
                f"{scores.detach().float().cpu().tolist()}"
            )
        return scores

    def _validate_processor_output(
        self,
        inputs: Mapping[str, torch.Tensor],
        name: str,
        expected_images: int,
    ) -> None:
        required = {"input_ids", "attention_mask"}
        if expected_images:
            required.update({"token_type_ids", "pixel_values"})
        missing = required - set(inputs)
        if missing:
            raise KeyError(f"Processor output for {name} is missing {sorted(missing)}.")
        sequence_length = inputs["input_ids"].shape[1]
        sequence_fields = ["attention_mask"]
        if "token_type_ids" in inputs:
            sequence_fields.append("token_type_ids")
        for field in sequence_fields:
            if inputs[field].shape != inputs["input_ids"].shape:
                raise ValueError(
                    f"{field} shape {tuple(inputs[field].shape)} does not match "
                    f"input_ids length {sequence_length} for {name}."
                )
        pixels = inputs.get("pixel_values")
        expected = (expected_images, 3, self.expected_image_size, self.expected_image_size)
        if expected_images and (pixels is None or tuple(pixels.shape) != expected):
            shape = None if pixels is None else tuple(pixels.shape)
            raise ValueError(
                f"Processor pixels for {name} have shape {shape}, expected {expected}."
            )


def _make_datamodule(config: Mapping[str, Any]) -> AmTICISDataModule:
    pipeline_config = build_pipeline_config({"data": config["data"]})
    module = AmTICISDataModule(config=pipeline_config)
    module.setup("fit")
    expected_frames = int(config["input"]["num_frames"])
    expected_size = int(config["input"]["image_size"])
    if module.config.data.num_frames != expected_frames:
        raise ValueError("Pipeline and input frame counts differ.")
    if module.config.data.image_size != expected_size:
        raise ValueError("Pipeline and MedGemma image sizes differ.")
    if module.config.views.active != list(config["input"]["views"]):
        raise ValueError("Pipeline and input view order differ.")
    return module


def _internal_splits(module: AmTICISDataModule, config: Mapping[str, Any]) -> dict[str, list[str]]:
    dataset = module._datasets["train"]
    splits = make_internal_split(
        dataset.samples,
        [dataset.label_at(index) for index in range(len(dataset))],
        tuning_fraction=float(config["selection"]["tuning_fraction"]),
        seed=int(config["seed"]),
    )
    val_names = list(module._datasets["val"].samples)
    from .common import assert_disjoint_studies

    assert_disjoint_studies({**splits, "final": val_names})
    manifest_path = REPO_ROOT / str(config["selection"]["manifest_path"])
    expected_manifest = {**splits, "final": val_names}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        if existing_manifest != expected_manifest:
            raise RuntimeError(
                f"Existing split manifest {manifest_path} differs from the deterministic split."
            )
    else:
        write_split_manifest(manifest_path, expected_manifest)
    return splits


def _loader_for_names(
    module: AmTICISDataModule,
    dataset_key: str,
    names: Sequence[str],
) -> DataLoader:
    dataset = module._datasets[dataset_key]
    name_set = set(names)
    indices = [index for index, name in enumerate(dataset.samples) if name in name_set]
    if len(indices) != len(name_set):
        raise ValueError("Requested study subset is not fully present in the dataset.")
    workers = module.config.loader.val_num_workers
    kwargs: dict[str, Any] = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": False,
        "drop_last": False,
    }
    if workers:
        kwargs.update(
            persistent_workers=module.config.loader.persistent_workers,
            prefetch_factor=module.config.loader.prefetch_factor,
        )
    return DataLoader(Subset(dataset, indices), **kwargs)


def _variants(config: Mapping[str, Any], phase: str) -> list[tuple[str, str]]:
    if phase == "tuning":
        return list(
            itertools.product(
                list(config["prompts"]),
                list(config["intensity_scaling"]),
            )
        )
    if phase == "final":
        selection_path = (
            REPO_ROOT
            / str(config["run"]["output_dir"])
            / str(config["run"]["name"])
            / "selection.yaml"
        )
        if not selection_path.exists():
            raise FileNotFoundError(
                f"Locked selection {selection_path} is missing; run --phase tuning first."
            )
        with open(selection_path, "r", encoding="utf-8") as handle:
            selected = yaml.safe_load(handle) or {}
        prompt_id = str(selected.get("prompt_id", ""))
        scaling_id = str(selected.get("scaling_id", ""))
        if prompt_id not in config["prompts"] or scaling_id not in config["intensity_scaling"]:
            raise ValueError(f"Invalid locked selection in {selection_path}.")
        return [(prompt_id, scaling_id)]
    selected = config["selection"]
    return [(str(selected["prompt_id"]), str(selected["scaling_id"]))]


def _balanced_smoke_names(
    module: AmTICISDataModule,
    names: Sequence[str],
    sample_count: int,
) -> list[str]:
    dataset = module._datasets["train"]
    allowed = set(names)
    by_class: dict[int, list[str]] = {0: [], 1: []}
    for index, name in enumerate(dataset.samples):
        if name in allowed:
            by_class[dataset.label_at(index)].append(name)
    per_class = max(sample_count // 2, 1)
    if any(len(values) < per_class for values in by_class.values()):
        raise ValueError("Internal tuning split is too small for a balanced smoke set.")
    selected = by_class[0][:per_class] + by_class[1][:per_class]
    return sorted(selected)


def evaluate_variant(
    *,
    config: Mapping[str, Any],
    scorer: MedGemmaScorer,
    loader: DataLoader,
    split_name: str,
    prompt_id: str,
    scaling_id: str,
    limit: int | None = None,
) -> dict[str, Any]:
    prompt = config["prompts"][prompt_id]
    scaling = config["intensity_scaling"][scaling_id]
    views = list(config["input"]["views"])
    threshold = float(config["evaluation"]["threshold"])
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    print(
        f"Evaluating {split_name}: prompt={prompt_id}, scaling={scaling_id}, "
        f"studies={len(loader.dataset)}",
        flush=True,
    )
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_index)

    for sample_index, batch in enumerate(loader):
        if limit is not None and sample_index >= limit:
            break
        name = batch["name"][0]
        per_view_images = [
            scan_to_rgb_frames(batch[view][0], scaling) for view in views
        ]
        scores = scorer.score(
            messages=build_messages(prompt, views, per_view_images),
            study_name=name,
        )
        rows.append(
            score_row(
                study_name=name,
                true_label=int(batch["label"].view(-1)[0].item()),
                scores=scores,
                model_name=str(config["model"]["pretrained_name"]),
                prompt_id=prompt_id,
                scaling_id=scaling_id,
                threshold=threshold,
                extra={
                    "num_frames": int(config["input"]["num_frames"]),
                    "views": "+".join(views),
                },
            )
        )
        completed = sample_index + 1
        if completed % 10 == 0 or completed == len(loader.dataset):
            print(f"  completed {completed}/{len(loader.dataset)} studies", flush=True)

    summary = summarize_rows(
        rows,
        threshold=threshold,
        bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["seed"]),
    )
    maximum_tie_rate = float(config["evaluation"]["maximum_tie_rate"])
    summary["tie_rate_exceeded"] = bool(summary["tie_rate"] > maximum_tie_rate)
    summary.update(
        {
            "split": split_name,
            "prompt_id": prompt_id,
            "scaling_id": scaling_id,
            "wall_seconds": float(time.perf_counter() - started),
            "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "peak_gpu_memory_bytes": int(
                max(
                    (torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())),
                    default=0,
                )
            ),
        }
    )
    artifact_split = f"{split_name}__{prompt_id}__{scaling_id}"
    write_artifacts(
        config=config,
        split_name=artifact_split,
        rows=rows,
        summary=summary,
        metadata=environment_metadata(str(config["model"]["revision"])),
    )
    return summary


def run_smoke_controls(
    config: Mapping[str, Any],
    scorer: MedGemmaScorer,
    module: AmTICISDataModule,
    names: Sequence[str],
) -> None:
    batch = next(iter(_loader_for_names(module, "train", names[:1])))
    name = batch["name"][0]
    prompt_id = str(config["selection"]["prompt_id"])
    scaling_id = str(config["selection"]["scaling_id"])
    prompt = config["prompts"][prompt_id]
    scaling = config["intensity_scaling"][scaling_id]
    views = list(config["input"]["views"])
    real_images = [scan_to_rgb_frames(batch[view][0], scaling) for view in views]
    reverse_images = [list(reversed(images)) for images in real_images]
    blank_images = [
        [Image.new("RGB", image.size, color=0) for image in images]
        for images in real_images
    ]
    scores = {
        "real": scorer.score(
            messages=build_messages(prompt, views, real_images), study_name=f"{name}:real"
        ),
        "reversed": scorer.score(
            messages=build_messages(prompt, views, reverse_images),
            study_name=f"{name}:reversed",
        ),
        "blank": scorer.score(
            messages=build_messages(prompt, views, blank_images), study_name=f"{name}:blank"
        ),
        "text_only": scorer.score(
            messages=build_messages(prompt, views, [[], []]),
            study_name=f"{name}:text_only",
            expected_images=0,
        ),
    }
    margins = {
        key: float((value[1] - value[0]).detach().float().cpu().item())
        for key, value in scores.items()
    }
    if margins["real"] in {margins["blank"], margins["text_only"]}:
        raise RuntimeError("Smoke controls show no score change from real DSA images.")
    output_dir = REPO_ROOT / str(config["run"]["output_dir"]) / str(config["run"]["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "smoke_controls.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump({"study": name, "score_margin": margins}, handle, sort_keys=False)


def run(config: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    validate_runtime()
    load_dotenv()
    seed_everything(int(config["seed"]))
    module = _make_datamodule(config)
    splits = _internal_splits(module, config)
    if phase == "final":
        names = module._datasets["val"].samples
        dataset_key = "val"
        split_name = "final"
        limit = None
    else:
        names = splits["tuning"]
        dataset_key = "train"
        split_name = phase
        if phase == "smoke":
            names = _balanced_smoke_names(
                module, names, int(config["evaluation"]["smoke_samples"])
            )
        limit = None

    variants = _variants(config, phase)
    scorer = MedGemmaScorer(config)
    if phase == "smoke":
        run_smoke_controls(config, scorer, module, names)
    summaries = []
    for prompt_id, scaling_id in variants:
        loader = _loader_for_names(module, dataset_key, names)
        summaries.append(
            evaluate_variant(
                config=config,
                scorer=scorer,
                loader=loader,
                split_name=split_name,
                prompt_id=prompt_id,
                scaling_id=scaling_id,
                limit=limit,
            )
        )
    if phase == "tuning":
        selected = max(
            summaries,
            key=lambda item: (item["metrics"]["f1_macro"], item["metrics"]["auroc"]),
        )
        selection_path = (
            REPO_ROOT
            / str(config["run"]["output_dir"])
            / str(config["run"]["name"])
            / "selection.yaml"
        )
        with open(selection_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "prompt_id": selected["prompt_id"],
                    "scaling_id": selected["scaling_id"],
                    "selection_metric": "f1_macro_then_auroc",
                    "tuning_metrics": selected["metrics"],
                },
                handle,
                sort_keys=False,
            )
    return summaries


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    summaries = run(config, args.phase)
    print(yaml.safe_dump(summaries, sort_keys=False))


if __name__ == "__main__":
    main()
