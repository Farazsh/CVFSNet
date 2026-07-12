import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from medgemma_binary.common import (
    assert_finite_scores,
    environment_metadata,
    load_config,
    make_internal_split,
    scan_to_rgb_frames,
    score_row,
)
from medgemma_binary.zero_shot import (
    answer_token_ids,
    build_messages,
    next_token_class_scores,
    selected_lm_head_scores,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    def encode(self, answer, add_special_tokens=False):
        del add_special_tokens
        return {"0": [10], "1": [11], "bad": [12, 13]}[answer]


class FakeProcessor:
    tokenizer = FakeTokenizer()


class CorrectedZeroShotTest(unittest.TestCase):
    def test_config_preserves_primary_input(self):
        config = load_config(REPO_ROOT / "medgemma_binary/config_zero_shot.yaml")

        self.assertEqual(config["input"]["num_frames"], 8)
        self.assertEqual(config["input"]["image_size"], 896)
        self.assertEqual(config["input"]["views"], ["AP", "sagittal"])
        self.assertEqual(config["model"]["dtype"], "bfloat16")
        self.assertTrue(config["model"]["revision"])

    def test_scan_conversion_preserves_temporal_order_and_rgb(self):
        clip = torch.stack(
            [torch.full((3, 4), value, dtype=torch.float32) for value in (0, 10, 20)]
        ).unsqueeze(0)
        frames = scan_to_rgb_frames(clip, {"mode": "minmax"})

        self.assertEqual(len(frames), 3)
        self.assertEqual(np.asarray(frames[0]).shape, (3, 4, 3))
        self.assertEqual(int(np.asarray(frames[0])[0, 0, 0]), 0)
        self.assertEqual(int(np.asarray(frames[-1])[0, 0, 0]), 255)
        self.assertTrue(np.array_equal(np.asarray(frames[1])[:, :, 0], np.asarray(frames[1])[:, :, 2]))

    def test_scan_conversion_rejects_invalid_or_degenerate_data(self):
        with self.assertRaises(ValueError):
            scan_to_rgb_frames(torch.zeros(1, 2, 3, 3), {"mode": "minmax"})
        invalid = torch.zeros(1, 2, 3, 3)
        invalid[0, 0, 0, 0] = torch.nan
        with self.assertRaises(FloatingPointError):
            scan_to_rgb_frames(invalid, {"mode": "minmax"})

    def test_messages_contain_all_images_in_stable_order(self):
        ap = [Image.new("RGB", (2, 2), color=index) for index in range(2)]
        sagittal = [Image.new("RGB", (2, 2), color=index + 2) for index in range(2)]
        prompt = {"preamble": "start", "question": "answer"}

        messages = build_messages(prompt, ["AP", "sagittal"], [ap, sagittal])
        content = messages[0]["content"]
        images = [item["image"] for item in content if item["type"] == "image"]
        text = [item["text"] for item in content if item["type"] == "text"]

        self.assertEqual(images, ap + sagittal)
        self.assertIn("FRAME 2 OF 2", text)
        self.assertEqual(text[-1], "answer")

    def test_answer_labels_must_be_distinct_single_tokens(self):
        self.assertEqual(answer_token_ids(FakeProcessor(), ["0", "1"]), [10, 11])
        with self.assertRaises(ValueError):
            answer_token_ids(FakeProcessor(), ["bad", "1"])
        with self.assertRaises(ValueError):
            answer_token_ids(FakeProcessor(), ["0", "0"])

    def test_next_token_scoring_uses_only_requested_vocab_entries(self):
        logits = torch.zeros(1, 1, 20)
        logits[0, 0, 10] = -2.0
        logits[0, 0, 11] = 3.0

        scores = next_token_class_scores(logits, [10, 11])

        torch.testing.assert_close(scores, torch.tensor([-2.0, 3.0]))
        with self.assertRaises(ValueError):
            next_token_class_scores(torch.zeros(1, 2, 20), [10, 11])

    def test_selected_lm_head_scores_accumulate_in_float32(self):
        hidden = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        weight = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=torch.bfloat16,
        )

        scores = selected_lm_head_scores(hidden, weight, [2, 3])

        self.assertEqual(scores.dtype, torch.float32)
        torch.testing.assert_close(scores, torch.tensor([11.0, 17.0]))

    def test_nonfinite_scores_fail_instead_of_becoming_predictions(self):
        with self.assertRaises(FloatingPointError):
            assert_finite_scores(torch.tensor([float("nan"), 0.0]), "study")
        with self.assertRaises(FloatingPointError):
            score_row(
                study_name="study",
                true_label=0,
                scores=torch.tensor([float("-inf"), 0.0]),
                model_name="model",
                prompt_id="prompt",
                scaling_id="scaling",
                threshold=0.5,
            )

    def test_internal_split_is_reproducible_stratified_and_disjoint(self):
        names = [f"{index:04d}_T0_C.dcm" for index in range(10)] + [
            f"{index + 10:04d}_T3_C.dcm" for index in range(10)
        ]
        labels = [0] * 10 + [1] * 10

        first = make_internal_split(names, labels, tuning_fraction=0.2, seed=14207)
        second = make_internal_split(names, labels, tuning_fraction=0.2, seed=14207)

        self.assertEqual(first, second)
        self.assertEqual(len(first["development"]), 16)
        self.assertEqual(len(first["tuning"]), 4)
        self.assertFalse(set(first["development"]) & set(first["tuning"]))
        self.assertEqual(sum("_T0_" in name for name in first["tuning"]), 2)

    def test_environment_metadata_is_yaml_serializable(self):
        import yaml

        yaml.safe_dump(environment_metadata("revision"))


class MedGemmaProcessorIntegrationTest(unittest.TestCase):
    def test_processor_expands_sixteen_images_to_native_shape(self):
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                "google/medgemma-1.5-4b-it",
                revision="91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
                local_files_only=True,
                use_fast=False,
            )
        except Exception as error:
            self.skipTest(f"Pinned MedGemma processor is not cached: {error}")

        frames = [Image.new("RGB", (896, 896), color=index) for index in range(16)]
        messages = build_messages(
            {"preamble": "Review DSA.", "question": "Reply 0 or 1."},
            ["AP", "sagittal"],
            [frames[:8], frames[8:]],
        )
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        self.assertEqual(tuple(inputs["pixel_values"].shape), (16, 3, 896, 896))
        self.assertEqual(inputs["attention_mask"].shape, inputs["input_ids"].shape)
        self.assertEqual(inputs["token_type_ids"].shape, inputs["input_ids"].shape)
        self.assertEqual(int((inputs["input_ids"] == 262144).sum().item()), 16 * 256)

    def test_medsiglip_processor_uses_native_shape_for_sixteen_images(self):
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                "google/medsiglip-448",
                revision="9cea28a1a1195f665105faa6e8544c112fd960a4",
                local_files_only=True,
                use_fast=False,
            )
        except Exception as error:
            self.skipTest(f"Pinned MedSigLIP processor is not cached: {error}")

        inputs = processor(
            text=["unsuccessful reperfusion", "successful reperfusion"],
            images=[Image.new("RGB", (448, 448), color=index) for index in range(16)],
            padding="max_length",
            return_tensors="pt",
        )

        self.assertEqual(tuple(inputs["pixel_values"].shape), (16, 3, 448, 448))
        self.assertEqual(inputs["input_ids"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
