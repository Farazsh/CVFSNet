"""Tests for the MedGemma QLoRA (E2) binary TICI experiment.

These cover the checks required by section 5.4 of the experiment plan plus a
regression test for the LoRA targeting bug: PEFT matches ``target_modules`` by
suffix, so bare leaf names such as ``q_proj`` silently inject adapters into the
SigLIP vision tower that the plan requires to stay frozen.
"""

import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from amticis_pipeline.transforms import ResizeView
from medgemma_binary.common import assert_disjoint_studies, load_config, scan_to_rgb_frames
from medgemma_binary.qlora import (
    balanced_subset,
    build_messages,
    resolve_target_modules,
    select_threshold,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "medgemma_binary/config_qlora_ap.yaml"


class FakeAttention(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(hidden, hidden))


class FakeMLP(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, nn.Linear(hidden, hidden))


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = FakeAttention(hidden)
        self.mlp = FakeMLP(hidden)


class FakeVisionAttention(nn.Module):
    """SigLIP names its output projection ``out_proj`` but shares q/k/v names."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, nn.Linear(hidden, hidden))


class FakeGemma3(nn.Module):
    """Mimics the real ``Gemma3ForConditionalGeneration`` module naming."""

    def __init__(self, layers: int = 2, hidden: int = 8) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            FakeDecoderLayer(hidden) for _ in range(layers)
        )
        self.model.vision_tower = nn.Module()
        self.model.vision_tower.vision_model = nn.Module()
        self.model.vision_tower.vision_model.encoder = nn.Module()
        self.model.vision_tower.vision_model.encoder.layers = nn.ModuleList(
            FakeVisionAttention(hidden) for _ in range(layers)
        )
        self.model.multi_modal_projector = nn.Module()
        self.model.multi_modal_projector.mm_input_projection = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, 4)


class LoraTargetingTest(unittest.TestCase):
    def test_language_scope_never_touches_the_vision_tower(self):
        targets = resolve_target_modules(FakeGemma3(), "all_linear_language")

        self.assertTrue(targets, "language scope matched nothing")
        for name in targets:
            self.assertNotIn("vision_tower", name)
            self.assertNotIn("multi_modal_projector", name)
            self.assertNotIn("lm_head", name)

    def test_targets_are_fully_qualified_not_bare_leaf_names(self):
        # A bare "q_proj" would also match the vision tower's q_proj by suffix.
        targets = resolve_target_modules(FakeGemma3(), "all_linear_language")

        for name in targets:
            self.assertIn(".", name, f"'{name}' is a bare leaf name and matches by suffix")
            self.assertTrue(name.startswith("model.language_model."))

    def test_language_scope_covers_all_seven_linears_per_layer(self):
        targets = resolve_target_modules(FakeGemma3(layers=3), "all_linear_language")

        self.assertEqual(len(targets), 3 * 7)

    def test_attention_only_scope_is_a_strict_subset(self):
        model = FakeGemma3()
        attention = resolve_target_modules(model, "attention_only")
        everything = resolve_target_modules(model, "all_linear_language")

        self.assertTrue(set(attention) < set(everything))
        self.assertTrue(all(name.rsplit(".", 1)[-1].endswith("_proj") for name in attention))
        self.assertFalse(any("mlp" in name for name in attention))

    def test_projector_scope_adds_the_projector_only(self):
        model = FakeGemma3()
        with_projector = set(resolve_target_modules(model, "all_linear_language_projector"))
        language = set(resolve_target_modules(model, "all_linear_language"))

        self.assertEqual(
            {name for name in with_projector - language if "multi_modal_projector" in name},
            with_projector - language,
        )
        self.assertFalse(any("vision_tower" in name for name in with_projector))

    def test_unknown_scope_raises(self):
        with self.assertRaises(ValueError):
            resolve_target_modules(FakeGemma3(), "everything_everywhere")


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_ap_only_at_medgemma_native_resolution(self):
        self.assertEqual(self.config["input"]["views"], ["AP"])
        self.assertEqual(self.config["input"]["image_size"], 896)
        self.assertEqual(self.config["data"]["pipeline_overrides"]["views"]["active"], ["AP"])

    def test_frame_count_has_one_source_of_truth(self):
        pipeline = self.config["data"]["pipeline_overrides"]["data"]

        self.assertEqual(self.config["input"]["num_frames"], pipeline["num_frames"])
        self.assertEqual(self.config["input"]["image_size"], pipeline["image_size"])
        self.assertEqual(pipeline["interpolation"], "trilinear")
        self.assertTrue(pipeline["align_corners"])
        self.assertEqual(pipeline["label_mode"], "binary")

    def test_published_qlora_recipe_is_preserved(self):
        # Google-Health/medgemma fine_tune_with_hugging_face.ipynb.
        self.assertEqual(self.config["quantization"]["bits"], 4)
        self.assertEqual(self.config["quantization"]["quant_type"], "nf4")
        self.assertTrue(self.config["quantization"]["double_quant"])
        self.assertEqual(self.config["quantization"]["compute_dtype"], "bfloat16")
        self.assertEqual(self.config["lora"]["rank"], 16)
        self.assertEqual(self.config["lora"]["alpha"], 16)
        self.assertAlmostEqual(self.config["lora"]["dropout"], 0.05)
        self.assertAlmostEqual(self.config["optimizer"]["adapter_lr"], 2e-4)
        self.assertAlmostEqual(self.config["training"]["gradient_clip_val"], 0.3)
        self.assertAlmostEqual(self.config["training"]["warmup_ratio"], 0.03)
        self.assertTrue(self.config["training"]["gradient_checkpointing"])

    def test_split_seed_is_independent_of_the_training_seed(self):
        # Every seed and strategy must share one development/tuning/final split.
        self.assertEqual(self.config["selection"]["split_seed"], 14207)
        self.assertEqual(
            self.config["selection"]["manifest_path"],
            "output_runs_lightning/medgemma_internal_split.json",
        )

    def test_training_transforms_are_frame_consistent_geometry_only(self):
        names = [
            step["name"]
            for step in self.config["data"]["pipeline_overrides"]["transforms"]["train"]
        ]

        self.assertEqual(names, ["RandomRotation", "Crop", "Resize"])
        # Intensity normalization happens in scan_to_rgb_frames, not the pipeline.
        self.assertEqual(self.config["data"]["pipeline_overrides"]["transforms"]["val"], [])


class PromptTest(unittest.TestCase):
    def setUp(self):
        self.prompt = load_config(CONFIG)["prompts"]["clinical_v1"]

    @staticmethod
    def _frames(count):
        return [Image.new("RGB", (8, 8)) for _ in range(count)]

    def test_frame_markers_come_from_the_data_not_the_template(self):
        for count in (4, 8, 16):
            content = build_messages(self.prompt, ["AP"], [self._frames(count)])[0]["content"]
            images = [block for block in content if block["type"] == "image"]
            markers = [block["text"] for block in content if block["type"] == "text"]

            self.assertEqual(len(images), count)
            self.assertIn(f"FRAME {count} OF {count}", markers)
            self.assertIn(f"FRAME 1 OF {count}", markers)

    def test_prompt_length_tracks_frame_count(self):
        short = build_messages(self.prompt, ["AP"], [self._frames(4)])[0]["content"]
        long = build_messages(self.prompt, ["AP"], [self._frames(16)])[0]["content"]

        self.assertLess(len(short), len(long))

    def test_views_are_labelled_and_ordered(self):
        content = build_messages(
            self.prompt, ["AP", "sagittal"], [self._frames(2), self._frames(2)]
        )[0]["content"]
        texts = [block["text"] for block in content if block["type"] == "text"]

        self.assertLess(texts.index("AP VIEW"), texts.index("SAGITTAL VIEW"))
        self.assertEqual(sum(block["type"] == "image" for block in content), 4)

    def test_prompt_never_leaks_a_label(self):
        text = f"{self.prompt['preamble']} {self.prompt['question']}"

        self.assertNotIn("answer is", text.lower())


class ResamplingTest(unittest.TestCase):
    def test_frame_count_and_size_are_configurable_without_code_edits(self):
        scan = np.random.rand(64, 48, 23).astype(np.float32)  # (H, W, T), odd T

        for frames in (4, 8, 16):
            clip = ResizeView(num_frames=frames, image_size=64)(scan)
            self.assertEqual(tuple(clip.shape), (1, frames, 64, 64))

    def test_views_are_resampled_independently(self):
        rng = np.random.default_rng(0)
        ap = rng.random((40, 40, 12)).astype(np.float32)
        sagittal = rng.random((30, 55, 19)).astype(np.float32)
        resize = ResizeView(num_frames=8, image_size=32)

        ap_clip, sagittal_clip = resize(ap), resize(sagittal)

        self.assertEqual(tuple(ap_clip.shape), tuple(sagittal_clip.shape))
        self.assertFalse(torch.allclose(ap_clip, sagittal_clip))

    def test_every_frame_becomes_three_channel_rgb_in_order(self):
        clip = torch.arange(8 * 4 * 4, dtype=torch.float32).reshape(1, 8, 4, 4)

        frames = scan_to_rgb_frames(clip, {"mode": "minmax"})

        self.assertEqual(len(frames), 8)
        for frame in frames:
            array = np.asarray(frame)
            self.assertEqual(array.shape, (4, 4, 3))
            self.assertTrue((array[..., 0] == array[..., 1]).all())
            self.assertTrue((array[..., 1] == array[..., 2]).all())
        # The clip increases monotonically over time, so frame order must too.
        means = [np.asarray(frame).mean() for frame in frames]
        self.assertEqual(means, sorted(means))

    def test_scaling_is_shared_across_frames_of_one_scan(self):
        # A bright late frame must not be renormalized away frame by frame.
        clip = torch.zeros(1, 4, 4, 4)
        clip[0, 3] = 100.0

        frames = scan_to_rgb_frames(clip, {"mode": "minmax"})

        self.assertEqual(np.asarray(frames[0]).max(), 0)
        self.assertEqual(np.asarray(frames[3]).max(), 255)

    def test_rejects_a_malformed_scan(self):
        with self.assertRaises(ValueError):
            scan_to_rgb_frames(torch.zeros(2, 4, 4, 4), {"mode": "minmax"})


class SelectionTest(unittest.TestCase):
    def test_splits_share_no_patient(self):
        with self.assertRaises(ValueError):
            assert_disjoint_studies(
                {"development": ["p1_a_C.dcm"], "tuning": ["p1_b_C.dcm"]}
            )
        assert_disjoint_studies({"development": ["p1_a_C.dcm"], "tuning": ["p2_b_C.dcm"]})

    def test_threshold_maximizes_macro_f1_on_the_tuning_split(self):
        rows = [
            {"true_label": 0, "probability_t2b3": 0.10},
            {"true_label": 0, "probability_t2b3": 0.20},
            {"true_label": 1, "probability_t2b3": 0.30},
            {"true_label": 1, "probability_t2b3": 0.40},
        ]

        threshold = select_threshold(rows)

        predictions = [int(row["probability_t2b3"] >= threshold) for row in rows]
        self.assertEqual(predictions, [0, 0, 1, 1])

    def test_threshold_falls_back_to_a_valid_probability(self):
        rows = [{"true_label": 1, "probability_t2b3": 0.9} for _ in range(3)]
        rows += [{"true_label": 0, "probability_t2b3": 0.1} for _ in range(3)]

        self.assertTrue(0.0 <= select_threshold(rows) <= 1.0)

    def test_balanced_subset_keeps_both_classes(self):
        names = [f"s{i}" for i in range(10)]
        labels = [0] * 5 + [1] * 5

        subset = balanced_subset(names, labels, 4)

        chosen = {name: label for name, label in zip(names, labels)}
        self.assertEqual(sorted(chosen[name] for name in subset), [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
