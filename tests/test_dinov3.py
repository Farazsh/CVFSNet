from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn as nn

from dinov3.dataloader import DINOv3TICIDataset, TemporalChannelTransform
from dinov3.evaluate import EXCEL_COLUMNS, load_processor, write_outputs
from dinov3.losses import build_loss
from dinov3.metrics import binary_metrics, tune_macro_f1_threshold
from dinov3.model import DINOv3BinaryClassifier


DATA_CONFIG = {
    "num_frames": 3,
    "image_size": 4,
    "interpolation": "trilinear",
    "align_corners": True,
    "percentile_lower": 0.0,
    "percentile_upper": 100.0,
    "split_name_extension": ".dcm",
    "file_extension": ".nii.gz",
    "view": {"name": "AP", "replace_from": "_C", "replace_to": "_C"},
    "augmentation": {
        "horizontal_flip_probability": 0.0,
        "rotation_probability": 0.0,
        "crop_scale": [1.0, 1.0],
        "crop_ratio": [1.0, 1.0],
    },
}


class DataLoaderTest(unittest.TestCase):
    def test_ap_volume_becomes_three_chronological_channels(self):
        scan = np.stack(
            [np.full((4, 4), frame, dtype=np.float32) for frame in range(5)],
            axis=-1,
        )
        dataset = DINOv3TICIDataset(
            ["0001_T2B_C_anon.dcm"],
            ".",
            DATA_CONFIG,
            image_mean=[0, 0, 0],
            image_std=[1, 1, 1],
            training=False,
        )
        with patch("dinov3.dataloader.load_nifti", return_value=scan) as loader:
            sample = dataset[0]
        self.assertEqual(tuple(sample["pixel_values"].shape), (3, 4, 4))
        self.assertTrue(torch.allclose(sample["pixel_values"][:, 0, 0], torch.tensor([0.0, 0.5, 1.0])))
        self.assertEqual(sample["label"].item(), 1)
        self.assertEqual(sample["raw_grade"], "T2B")
        self.assertTrue(str(loader.call_args.args[0]).endswith("0001_T2B_C_anon.nii.gz"))

    def test_degenerate_scan_is_rejected(self):
        dataset = DINOv3TICIDataset(
            ["0001_T0_C_anon.dcm"], ".", DATA_CONFIG, [0, 0, 0], [1, 1, 1], False
        )
        with patch("dinov3.dataloader.load_nifti", return_value=np.ones((4, 4, 5), np.float32)):
            with self.assertRaisesRegex(ValueError, "degenerate"):
                dataset[0]

    def test_geometry_is_shared_across_channels(self):
        transform = TemporalChannelTransform(
            8,
            {
                "horizontal_flip_probability": 1.0,
                "rotation_probability": 1.0,
                "rotation_degrees": 5.0,
                "crop_scale": [0.9, 0.9],
                "crop_ratio": [1.0, 1.0],
            },
        )
        image = torch.arange(64, dtype=torch.float32).view(1, 8, 8).repeat(3, 1, 1)
        output = transform(image)
        self.assertTrue(torch.equal(output[0], output[1]))
        self.assertTrue(torch.equal(output[1], output[2]))


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(hidden_size=2, num_register_tokens=4, patch_size=16)

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        cls = torch.ones(batch, 1, 2)
        registers = torch.full((batch, 4, 2), 100.0)
        patches = torch.full((batch, 4, 2), 3.0)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, registers, patches], dim=1))


class FakeLightlyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.embed_dim = 2
        self.patch_size = 16
        self.n_storage_tokens = 4

    def forward(self, pixel_values, is_training=False):
        self.last_is_training = is_training
        batch = pixel_values.shape[0]
        return {
            "x_norm_clstoken": torch.ones(batch, 2),
            "x_norm_patchtokens": torch.full((batch, 4, 2), 3.0),
        }


class ModelTest(unittest.TestCase):
    def test_classifier_excludes_register_tokens(self):
        model = DINOv3BinaryClassifier("unused", backbone=FakeBackbone(), dropout=0.0)
        head = nn.Linear(4, 1, bias=False)
        nn.init.ones_(head.weight)
        model.classifier = head
        result = model(torch.zeros(2, 3, 32, 32))
        self.assertTrue(torch.equal(result, torch.tensor([8.0, 8.0])))

    def test_invalid_shape_is_rejected(self):
        model = DINOv3BinaryClassifier("unused", backbone=FakeBackbone())
        with self.assertRaisesRegex(ValueError, "Expected"):
            model(torch.zeros(2, 1, 32, 32))

    def test_lightly_backend_uses_normalized_cls_and_patch_tokens(self):
        backbone = FakeLightlyBackbone()
        model = DINOv3BinaryClassifier(
            "unused", backend="lightly", lightly_name="dinov3/vits16", backbone=backbone,
            dropout=0.0,
        )
        head = nn.Linear(4, 1, bias=False)
        nn.init.ones_(head.weight)
        model.classifier = head
        result = model(torch.zeros(2, 3, 32, 32))
        self.assertTrue(torch.equal(result, torch.tensor([8.0, 8.0])))
        self.assertTrue(backbone.last_is_training)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            DINOv3BinaryClassifier("unused", backend="unknown", backbone=FakeBackbone())


class BackendConfigTest(unittest.TestCase):
    def test_lightly_normalization_does_not_require_huggingface(self):
        processor, mean, std = load_processor(
            {
                "model": {
                    "backend": "lightly",
                    "lightly_normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                }
            },
            token=None,
        )
        self.assertIsNone(processor)
        self.assertEqual(mean, [0.485, 0.456, 0.406])
        self.assertEqual(std, [0.229, 0.224, 0.225])


class MetricsAndLossTest(unittest.TestCase):
    def test_reference_metrics(self):
        metrics = binary_metrics([0.1, 0.4, 0.6, 0.9], [0, 1, 0, 1], threshold=0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertEqual((metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]), (1, 1, 1, 1))

    def test_threshold_tuning_is_deterministic(self):
        threshold, score = tune_macro_f1_threshold([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
        self.assertEqual(threshold, 0.5)
        self.assertEqual(score, 1.0)

    def test_unweighted_bce_is_default(self):
        loss = build_loss({"name": "bce_with_logits", "pos_weight": None})
        self.assertIsNone(loss.pos_weight)


class ExportTest(unittest.TestCase):
    def test_workbook_uses_reference_column_order(self):
        config = {
            "seed": 14207,
            "model": {"pretrained_name": "facebook/dinov3-vits16-pretrain-lvd1689m"},
            "evaluation": {
                "scores_filename": "scores.csv",
                "json_filename": "evaluation.json",
                "workbook_suffix": "eval.xlsx",
                "excel_metadata": {
                    "Model": "dinov3-vits16",
                    "Dimension": "2D",
                    "Label1": "BinaryTICI",
                    "Label2": "T012a_vs_T2b3",
                    "InputSize": "3x224x224",
                    "Optimizer": "AdamW",
                    "Loop": "train_val",
                    "ExperimentVersion": "seed14207",
                    "ValidationSplit": "val",
                },
            },
        }
        predictions = {
            "names": ["a", "b"],
            "raw_grades": ["T0", "T3"],
            "targets": np.array([0, 1]),
            "logits": np.array([-1.0, 1.0]),
            "probabilities": np.array([0.2, 0.8]),
        }
        metrics = binary_metrics(predictions["probabilities"], predictions["targets"], 0.5)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_outputs(
                Path(directory), config, Path(directory) / "best.ckpt", predictions,
                0.5, 1.0, metrics, 3, "revision",
            )
            with ZipFile(paths["workbook"]) as workbook:
                shared = workbook.read("xl/sharedStrings.xml").decode("utf-8")
            positions = [shared.index(f">{column}<") for column in EXCEL_COLUMNS]
            self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
