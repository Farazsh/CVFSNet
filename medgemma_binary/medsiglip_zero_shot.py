"""MedSigLIP zero-shot control for the corrected binary TICI experiment."""

from __future__ import annotations

import argparse
import os
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from PIL import Image

from .common import (
    REPO_ROOT,
    environment_metadata,
    load_config,
    load_dotenv,
    scan_to_rgb_frames,
    score_row,
    seed_everything,
    summarize_rows,
    write_artifacts,
)
from .zero_shot import (
    _balanced_smoke_names,
    _internal_splits,
    _loader_for_names,
    _make_datamodule,
    _variants,
    validate_runtime,
)


DEFAULT_CONFIG = Path(__file__).with_name("config_medsiglip_zero_shot.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("smoke", "tuning", "final"), default="smoke")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


class MedSigLIPScorer:
    def __init__(self, config: Mapping[str, Any]) -> None:
        from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

        model_cfg = config["model"]
        revision = str(model_cfg.get("revision") or "")
        if not revision:
            raise ValueError("model.revision must be pinned before evaluation.")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        kwargs = {"revision": revision, "token": token}
        self.processor = AutoProcessor.from_pretrained(
            model_cfg["pretrained_name"],
            **kwargs,
            use_fast=bool(model_cfg.get("use_fast_processor", False)),
        )
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(
            model_cfg["pretrained_name"],
            **kwargs,
            dtype=torch.bfloat16,
            device_map=model_cfg.get("device_map", "auto"),
            low_cpu_mem_usage=True,
        ).eval()
        self.device = self.model.device
        self.num_frames = int(config["input"]["num_frames"])
        self.expected_images = self.num_frames * len(config["input"]["views"])
        self.expected_size = int(config["input"]["image_size"])

    @torch.inference_mode()
    def score(
        self,
        *,
        images: Sequence[Image.Image],
        text_labels: Sequence[str],
        study_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(text_labels) != 2:
            raise ValueError("MedSigLIP requires exactly two class descriptions.")
        inputs = self.processor(
            text=list(text_labels),
            images=list(images),
            padding="max_length",
            return_tensors="pt",
        )
        pixels = inputs.get("pixel_values")
        expected = (self.expected_images, 3, self.expected_size, self.expected_size)
        if pixels is None or tuple(pixels.shape) != expected:
            shape = None if pixels is None else tuple(pixels.shape)
            raise ValueError(f"Processor pixels for {study_name} are {shape}, expected {expected}.")
        if inputs["input_ids"].shape[0] != 2:
            raise ValueError("MedSigLIP processor did not produce two text inputs.")
        inputs = inputs.to(self.device, dtype=torch.bfloat16)
        logits = self.model(**inputs).logits_per_image.float()
        if tuple(logits.shape) != (self.expected_images, 2):
            raise ValueError(
                f"MedSigLIP logits for {study_name} are {tuple(logits.shape)}, "
                f"expected {(self.expected_images, 2)}."
            )
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite MedSigLIP logits for {study_name}.")
        aggregate = logits.mean(dim=0)
        ap = logits[: self.num_frames].mean(dim=0)
        sagittal = logits[self.num_frames :].mean(dim=0)
        return aggregate, ap, sagittal


def evaluate_variant(
    *,
    config: Mapping[str, Any],
    scorer: MedSigLIPScorer,
    loader: Any,
    split_name: str,
    prompt_id: str,
    scaling_id: str,
    limit: int | None = None,
) -> dict[str, Any]:
    text_labels = config["prompts"][prompt_id]["text_labels"]
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
        images = [
            image
            for view in views
            for image in scan_to_rgb_frames(batch[view][0], scaling)
        ]
        scores, ap_scores, sagittal_scores = scorer.score(
            images=images,
            text_labels=text_labels,
            study_name=name,
        )
        ap_probability = float(torch.softmax(ap_scores, dim=0)[1].item())
        sagittal_probability = float(torch.softmax(sagittal_scores, dim=0)[1].item())
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
                    "probability_t2b3_ap": ap_probability,
                    "probability_t2b3_sagittal": sagittal_probability,
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
            "aggregation": "mean_frame_logits",
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
    write_artifacts(
        config=config,
        split_name=f"{split_name}__{prompt_id}__{scaling_id}",
        rows=rows,
        summary=summary,
        metadata=environment_metadata(str(config["model"]["revision"])),
    )
    return summary


def run_smoke_controls(
    config: Mapping[str, Any],
    scorer: MedSigLIPScorer,
    module: Any,
    names: Sequence[str],
) -> None:
    batch = next(iter(_loader_for_names(module, "train", names[:1])))
    name = batch["name"][0]
    prompt_id = str(config["selection"]["prompt_id"])
    scaling_id = str(config["selection"]["scaling_id"])
    text_labels = config["prompts"][prompt_id]["text_labels"]
    scaling = config["intensity_scaling"][scaling_id]
    views = list(config["input"]["views"])
    per_view = [scan_to_rgb_frames(batch[view][0], scaling) for view in views]
    variants = {
        "real": [image for images in per_view for image in images],
        "reversed": [image for images in per_view for image in reversed(images)],
        "blank": [
            Image.new("RGB", image.size, color=0)
            for images in per_view
            for image in images
        ],
    }
    margins: dict[str, float] = {}
    for variant, images in variants.items():
        scores, _, _ = scorer.score(
            images=images,
            text_labels=text_labels,
            study_name=f"{name}:{variant}",
        )
        margins[variant] = float((scores[1] - scores[0]).cpu().item())
    if margins["real"] == margins["blank"]:
        raise RuntimeError("MedSigLIP smoke control shows no effect from real DSA images.")
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
        dataset_key, split_name, limit = "val", "final", None
    else:
        names = splits["tuning"]
        dataset_key, split_name = "train", phase
        if phase == "smoke":
            names = _balanced_smoke_names(
                module, names, int(config["evaluation"]["smoke_samples"])
            )
        limit = None

    variants = _variants(config, phase)
    scorer = MedSigLIPScorer(config)
    if phase == "smoke":
        run_smoke_controls(config, scorer, module, names)
    summaries = []
    for prompt_id, scaling_id in variants:
        summaries.append(
            evaluate_variant(
                config=config,
                scorer=scorer,
                loader=_loader_for_names(module, dataset_key, names),
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
    print(yaml.safe_dump(run(config, args.phase), sort_keys=False))


if __name__ == "__main__":
    main()
