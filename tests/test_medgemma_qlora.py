"""Tests for the MedGemma QLoRA (E2) binary TICI experiment.

These cover the checks required by section 5.4 of the experiment plan plus a
regression test for the LoRA targeting bug: PEFT matches ``target_modules`` by
suffix, so bare leaf names such as ``q_proj`` silently inject adapters into the
SigLIP vision tower that the plan requires to stay frozen.
"""

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from amticis_pipeline.transforms import ResizeView
from medgemma_binary.common import (
    apply_overrides,
    assert_disjoint_studies,
    load_config,
    scan_to_rgb_frames,
)
from medgemma_binary.data import MedGemmaData
from medgemma_binary.qlora import (
    balanced_subset,
    build_messages,
    eval_split_label,
    resolve_monitor,
    resolve_target_modules,
    select_threshold,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "medgemma_binary/config_qlora_ap.yaml"
VAL_CONFIGS = {
    "ap": REPO_ROOT / "medgemma_binary/config_qlora_val_ap.yaml",
    "sag": REPO_ROOT / "medgemma_binary/config_qlora_val_sag.yaml",
    "dual": REPO_ROOT / "medgemma_binary/config_qlora_val_dual.yaml",
}


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


# --------------------------------------------------------------------------- #
# Wave 2: the full_train_val protocol
# --------------------------------------------------------------------------- #
class FakeDataset:
    """Stands in for an AmTICIS dataset: names, labels, and nothing else."""

    def __init__(self, names, labels) -> None:
        self.samples = list(names)
        self._labels = [int(label) for label in labels]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {"name": self.samples[index], "label": self._labels[index]}

    def label_at(self, index):
        return self._labels[index]


class FakeLoaderConfig:
    num_workers = 0
    prefetch_factor = 2


class FakePipelineConfig:
    loader = FakeLoaderConfig()


class FakeModule:
    def __init__(self, train: FakeDataset, val: FakeDataset) -> None:
        self._datasets = {"train": train, "val": val}
        self.config = FakePipelineConfig()


def _study_names(prefix: str, count: int) -> list[str]:
    # patient_id() takes everything before the first underscore, so a distinct
    # prefix per study keeps the patients disjoint.
    return [f"{prefix}{index:03d}_a_C.nii.gz" for index in range(count)]


def _alternating(count: int) -> list[int]:
    return [index % 2 for index in range(count)]


class ProtocolTest(unittest.TestCase):
    """The split protocol that puts MedGemma on the VideoMAE/CVFSNet footing."""

    TRAIN, VAL = 261, 150

    def setUp(self):
        self.train_names = _study_names("p", self.TRAIN)
        self.val_names = _study_names("q", self.VAL)
        self.train_dataset = FakeDataset(self.train_names, _alternating(self.TRAIN))
        self.val_dataset = FakeDataset(self.val_names, _alternating(self.VAL))
        self._tmp = __import__("tempfile").TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _build(self, protocol: str, config_key: str = "ap") -> MedGemmaData:
        config = load_config(VAL_CONFIGS[config_key])
        config["selection"]["protocol"] = protocol
        config["selection"]["manifest_path"] = str(
            Path(self._tmp.name) / f"{protocol}_split.json"
        )
        module = FakeModule(self.train_dataset, self.val_dataset)
        with mock.patch("medgemma_binary.data._pipeline_module", return_value=module):
            return MedGemmaData(config)

    def test_full_train_val_trains_on_every_train_study_and_stops_on_val(self):
        data = self._build("full_train_val")

        self.assertEqual(len(data.development), self.TRAIN)
        self.assertEqual(len(data.tuning), self.VAL)
        self.assertEqual(len(data.final), self.VAL)
        self.assertEqual(data.development, sorted(self.train_names))
        # tuning deliberately aliases final: the run selects on what it reports.
        self.assertEqual(data.tuning, data.final)

    def test_full_train_val_never_evaluates_on_a_trained_patient(self):
        data = self._build("full_train_val")

        assert_disjoint_studies({"development": data.development, "final": data.final})
        self.assertFalse(set(data.development) & set(data.final))

    def test_internal_protocol_is_unchanged(self):
        # Regression guard: wave 1's 208/53/150 split must not move.
        data = self._build("internal")

        self.assertEqual(len(data.development), 208)
        self.assertEqual(len(data.tuning), 53)
        self.assertEqual(len(data.final), self.VAL)
        self.assertFalse(set(data.development) & set(data.tuning))

    def test_protocol_defaults_to_internal(self):
        config = load_config(CONFIG)

        self.assertNotIn("protocol", config["selection"])

        config["selection"]["manifest_path"] = str(Path(self._tmp.name) / "default.json")
        module = FakeModule(self.train_dataset, self.val_dataset)
        with mock.patch("medgemma_binary.data._pipeline_module", return_value=module):
            data = MedGemmaData(config)

        self.assertEqual(data.protocol, "internal")
        self.assertEqual(len(data.tuning), 53)

    def test_unknown_protocol_raises(self):
        with self.assertRaises(ValueError):
            self._build("train_on_everything")

    def test_labels_for_resolves_val_names_under_full_train_val(self):
        # data.tuning holds val names here; an index over the train dataset alone
        # would raise KeyError, and --limit calls this on every smoke run.
        data = self._build("full_train_val")

        labels = data.labels_for(data.tuning)

        self.assertEqual(len(labels), self.VAL)
        expected = {name: index % 2 for index, name in enumerate(self.val_names)}
        self.assertEqual(labels, [expected[name] for name in data.tuning])

    def test_labels_for_still_resolves_train_names(self):
        data = self._build("full_train_val")

        labels = data.labels_for(data.development)

        self.assertEqual(len(labels), self.TRAIN)

    def test_labels_for_rejects_an_unknown_study(self):
        data = self._build("full_train_val")

        with self.assertRaises(KeyError):
            data.labels_for(["not_a_study_C.nii.gz"])

    def test_tuning_loader_reads_the_val_dataset_under_full_train_val(self):
        data = self._build("full_train_val")

        loader = data.loader("tuning", shuffle=False, num_workers=0)

        self.assertIs(loader.dataset.dataset, self.val_dataset)
        self.assertEqual(len(loader.dataset), self.VAL)

    def test_tuning_loader_reads_the_train_dataset_under_internal(self):
        data = self._build("internal")

        loader = data.loader("tuning", shuffle=False, num_workers=0)

        self.assertIs(loader.dataset.dataset, self.train_dataset)
        self.assertEqual(len(loader.dataset), 53)

    def test_development_loader_stays_on_the_augmented_module(self):
        # Both modules are the same fake here, so assert on the dataset key instead:
        # development must resolve entirely within the train dataset.
        data = self._build("full_train_val")

        loader = data.loader("development", shuffle=True, num_workers=0)

        self.assertEqual(len(loader.dataset), self.TRAIN)

    def test_manifest_records_the_new_split_and_is_reconciled(self):
        import json

        data = self._build("full_train_val")
        path = Path(self._tmp.name) / "full_train_val_split.json"

        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)

        self.assertEqual(len(manifest["development"]), self.TRAIN)
        self.assertEqual(len(manifest["tuning"]), self.VAL)
        self.assertEqual(len(manifest["final"]), self.VAL)

        # A second run against the same manifest must agree ...
        self._build("full_train_val")
        # ... and a disagreeing one must fail loudly rather than re-split.
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({**manifest, "development": manifest["development"][:10]}, handle)
        with self.assertRaises(RuntimeError):
            self._build("full_train_val")

        self.assertEqual(data.protocol, "full_train_val")

    def test_the_wave_one_manifest_is_a_different_file(self):
        # A 261/150/150 manifest can never equal the 208/53/150 one, so the two
        # protocols must not share a manifest path or _reconcile_manifest raises.
        wave_one = load_config(CONFIG)["selection"]["manifest_path"]
        for key in VAL_CONFIGS:
            wave_two = load_config(VAL_CONFIGS[key])["selection"]["manifest_path"]
            self.assertNotEqual(wave_one, wave_two)
            self.assertEqual(
                wave_two, "output_runs_lightning/medgemma_full_train_val_split.json"
            )


class MonitorTest(unittest.TestCase):
    def test_monitor_defaults_to_the_wave_one_behaviour(self):
        self.assertEqual(resolve_monitor({}), "f1_macro_then_auroc")
        self.assertEqual(
            resolve_monitor(load_config(CONFIG)["training"]), "f1_macro_then_auroc"
        )

    def test_the_val_configs_monitor_auroc(self):
        for key, path in VAL_CONFIGS.items():
            with self.subTest(run=key):
                self.assertEqual(resolve_monitor(load_config(path)["training"]), "auroc")

    def test_unknown_monitor_raises(self):
        with self.assertRaises(ValueError):
            resolve_monitor({"monitor": "loss"})

    def test_auroc_monitor_picks_the_max_auroc_epoch_over_the_max_f1_one(self):
        # Reproduces the selection comparison in train(): epoch 2 wins on tuned
        # macro F1, epoch 1 wins on AUROC. Under monitor: auroc, epoch 1 must win.
        epochs = [
            {"epoch": 1, "f1": 0.60, "auroc": 0.90},
            {"epoch": 2, "f1": 0.75, "auroc": 0.80},
        ]

        for monitor, expected in (("auroc", 1), ("f1_macro_then_auroc", 2)):
            with self.subTest(monitor=monitor):
                best = {"macro_f1": -1.0, "auroc": -1.0, "epoch": -1}
                for record in epochs:
                    if monitor == "auroc":
                        improved = record["auroc"] > best["auroc"]
                    else:
                        improved = (record["f1"], record["auroc"]) > (
                            best["macro_f1"],
                            best["auroc"],
                        )
                    if improved:
                        best = {
                            "macro_f1": record["f1"],
                            "auroc": record["auroc"],
                            "epoch": record["epoch"],
                        }

                self.assertEqual(best["epoch"], expected)


class ValConfigTest(unittest.TestCase):
    IMAGES = {"ap": 8, "sag": 8, "dual": 16}
    VIEWS = {"ap": ["AP"], "sag": ["sagittal"], "dual": ["AP", "sagittal"]}

    def test_images_per_study_is_8_8_16(self):
        for key, expected in self.IMAGES.items():
            with self.subTest(run=key):
                config = load_config(VAL_CONFIGS[key])
                images = int(config["input"]["num_frames"]) * len(config["input"]["views"])

                self.assertEqual(images, expected)
                self.assertEqual(config["input"]["views"], self.VIEWS[key])

    def test_the_two_view_declarations_agree(self):
        # data.py cross-checks these and raises if they disagree; a config that
        # sets only one of them would train on the wrong view.
        for key in VAL_CONFIGS:
            with self.subTest(run=key):
                config = load_config(VAL_CONFIGS[key])

                self.assertEqual(
                    config["input"]["views"],
                    config["data"]["pipeline_overrides"]["views"]["active"],
                )

    def test_mismatched_views_raise(self):
        from types import SimpleNamespace

        from medgemma_binary import data as data_module

        config = load_config(VAL_CONFIGS["ap"])
        config["input"]["views"] = ["AP", "sagittal"]  # pipeline still says [AP]
        resolved = SimpleNamespace(
            data=SimpleNamespace(num_frames=8, image_size=896, label_mode="binary"),
            views=SimpleNamespace(active=["AP"]),
        )
        fake = mock.MagicMock()
        fake.config = resolved

        with mock.patch.object(data_module, "AmTICISDataModule", return_value=fake):
            with self.assertRaises(ValueError):
                data_module._pipeline_module(config, augment=False)

    def test_run_names_and_labels_do_not_collide_with_wave_one(self):
        wave_one = load_config(CONFIG)["run"]["name"]
        names = {load_config(path)["run"]["name"] for path in VAL_CONFIGS.values()}

        self.assertEqual(len(names), 3)
        self.assertNotIn(wave_one, names)
        for path in VAL_CONFIGS.values():
            self.assertEqual(eval_split_label(load_config(path)), "val")

    def test_eval_split_label_defaults_to_tuning(self):
        self.assertEqual(eval_split_label(load_config(CONFIG)), "tuning")
        self.assertEqual(eval_split_label({"run": {}}), "tuning")

    def test_the_qlora_recipe_is_carried_over_unchanged(self):
        wave_one = load_config(CONFIG)
        for key, path in VAL_CONFIGS.items():
            with self.subTest(run=key):
                config = load_config(path)
                for section in ("quantization", "lora", "optimizer", "model"):
                    self.assertEqual(config[section], wave_one[section])
                for field in (
                    "micro_batch_size",
                    "gradient_accumulation_steps",
                    "max_epochs",
                    "early_stopping_patience",
                    "warmup_ratio",
                    "gradient_clip_val",
                    "gradient_checkpointing",
                ):
                    self.assertEqual(
                        config["training"][field], wave_one["training"][field], field
                    )
                self.assertEqual(config["seed"], 14207)
                self.assertEqual(config["selection"]["prompt_id"], "clinical_v1")
                self.assertEqual(config["selection"]["scaling_id"], "percentile_1_99")
                self.assertFalse(
                    config["data"]["pipeline_overrides"]["loader"]["use_weighted_sampler"]
                )


class SmokeOverrideTest(unittest.TestCase):
    def test_explicit_set_beats_the_smoke_defaults(self):
        # main() applies the smoke defaults and *then* the --set overrides, so
        # --set training.max_epochs=3 must survive. The old order discarded it.
        config = load_config(VAL_CONFIGS["ap"])
        config["training"]["max_epochs"] = 1
        config["training"]["early_stopping_patience"] = 1
        config["evaluation"]["bootstrap_replicates"] = 100

        apply_overrides(config, ["training.max_epochs=3"])

        self.assertEqual(config["training"]["max_epochs"], 3)
        # Untouched smoke defaults still apply.
        self.assertEqual(config["training"]["early_stopping_patience"], 1)
        self.assertEqual(config["evaluation"]["bootstrap_replicates"], 100)

    def test_overrides_do_not_disturb_the_rest_of_the_config(self):
        config = load_config(VAL_CONFIGS["dual"])

        apply_overrides(config, ["training.max_epochs=3"])

        self.assertEqual(config["input"]["views"], ["AP", "sagittal"])
        self.assertEqual(config["selection"]["protocol"], "full_train_val")


if __name__ == "__main__":
    unittest.main()
