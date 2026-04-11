from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.apps.adversarial_preview import (
    build_sample_delta_visualization,
    generate_feature_preview,
    build_layer_visualization,
    discover_attack_config_options,
    discover_checkpoint_options,
    infer_model_family_from_checkpoint,
    ordered_feature_layer_names,
    FeaturePreviewResult,
)
from src.attacks import AttackConfig
from src.models.backbones.convnext import ConvNeXt
from src.models.base import TorchSegmentationModelAdapter


class AdversarialPreviewDiscoveryTest(unittest.TestCase):
    def test_infer_model_family_from_checkpoint_name(self) -> None:
        self.assertEqual(infer_model_family_from_checkpoint("models/UperNet_ConvNext_T_VOC_clean.pth"), "upernet_convnext")
        self.assertEqual(infer_model_family_from_checkpoint("models/UperNet_ResNet50_VOC_clean.pth"), "upernet_resnet50")
        self.assertEqual(infer_model_family_from_checkpoint("models/Segmenter_ViT_S_VOC_clean.pth"), "segmenter_vit_s")
        self.assertIsNone(infer_model_family_from_checkpoint("models/custom_unknown_model.pth"))

    def test_discover_attack_config_options_skips_non_attack_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "fgsm.yaml").write_text("name: fgsm\nepsilon: 0.1\nsteps: 1\n", encoding="utf-8")
            (root / "eval.yaml").write_text("report_name: demo\n", encoding="utf-8")

            options = discover_attack_config_options(root)

            self.assertEqual(len(options), 1)
            self.assertEqual(options[0].attack_name, "fgsm")

    def test_discover_checkpoint_options_includes_inferred_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "UperNet_ConvNext_T_custom.pth").write_text("placeholder", encoding="utf-8")
            (root / "ignore.txt").write_text("placeholder", encoding="utf-8")

            options = discover_checkpoint_options(root, include_known=False)

            self.assertEqual(len(options), 1)
            self.assertEqual(options[0].family, "upernet_convnext")


class AdversarialPreviewVisualizationTest(unittest.TestCase):
    def test_ordered_feature_layer_names_prefers_stage_and_block_layers(self) -> None:
        features = {
            "encoder:last": torch.randn(1, 4, 2, 2),
            "encoder:block01": torch.randn(1, 4, 2, 2),
            "encoder:block00": torch.randn(1, 4, 2, 2),
        }

        ordered = ordered_feature_layer_names(features)

        self.assertEqual(ordered, ["encoder:block00", "encoder:block01"])

    def test_ordered_feature_layer_names_prefers_backbone_blocks_over_stage_summaries(self) -> None:
        features = {
            "backbone:stage0": torch.randn(1, 4, 2, 2),
            "backbone:stage0:block01": torch.randn(1, 4, 2, 2),
            "backbone:stage0:block00": torch.randn(1, 4, 2, 2),
        }

        ordered = ordered_feature_layer_names(features)

        self.assertEqual(ordered, ["backbone:stage0:block00", "backbone:stage0:block01"])

    def test_convnext_forward_features_can_collect_block_intermediates(self) -> None:
        backbone = ConvNeXt("T_CVST")
        inputs = torch.randn(1, 3, 64, 64)

        stage_outputs, intermediates = backbone.forward_features(inputs, collect_intermediates=True)

        self.assertEqual(len(stage_outputs), 4)
        self.assertIn("backbone:stage0:block00", intermediates)
        self.assertIn("backbone:stage3:block02", intermediates)

    def test_generate_feature_preview_skips_attack_when_radius_is_zero(self) -> None:
        model = nn.Conv2d(1, 2, kernel_size=1, bias=True)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[[[3.0]]], [[[-3.0]]]]))
            model.bias.copy_(torch.tensor([-0.5, 0.5]))
        adapter = TorchSegmentationModelAdapter(model=model, num_classes=2, device="cpu")
        image = torch.ones(1, 4, 4, dtype=torch.float32) * 0.75
        target = torch.zeros(4, 4, dtype=torch.long)
        attack_config = AttackConfig(name="fgsm", epsilon=2.0 / 255.0, step_size=2.0 / 255.0, steps=1).with_radius_255(0)

        result = generate_feature_preview(adapter, attack_config, image, target, sample_id="sample_zero")

        self.assertEqual(float(result.epsilon), 0.0)
        self.assertTrue(result.attack_metadata["skipped"])
        self.assertEqual(result.attack_metadata["reason"], "zero_radius")
        self.assertTrue((result.clean_image == result.adversarial_image).all())
        self.assertEqual(float(result.sample_delta_mean), 0.0)
        self.assertEqual(float(result.sample_delta_max), 0.0)

    def test_build_layer_visualization_returns_displayable_images_and_stats(self) -> None:
        result = FeaturePreviewResult(
            sample_id="sample_0001",
            attack_name="fgsm",
            epsilon=0.1,
            step_size=0.1,
            steps=1,
            clean_image=torch.zeros(8, 8, 3, dtype=torch.uint8).numpy(),
            adversarial_image=torch.zeros(8, 8, 3, dtype=torch.uint8).numpy(),
            perturbation_image=torch.zeros(8, 8, 3, dtype=torch.uint8).numpy(),
            sample_delta_heatmap=torch.zeros(8, 8, 3, dtype=torch.uint8).numpy(),
            sample_delta_mean=0.0,
            sample_delta_max=0.0,
            ground_truth=torch.zeros(8, 8, dtype=torch.int64).numpy(),
            clean_prediction=torch.zeros(8, 8, dtype=torch.int64).numpy(),
            adversarial_prediction=torch.ones(8, 8, dtype=torch.int64).numpy(),
            clean_features={"encoder:block00": torch.randn(1, 4, 4, 4)},
            adversarial_features={"encoder:block00": torch.randn(1, 4, 4, 4)},
            layer_names=["encoder:block00"],
            attack_metadata={},
        )

        payload = build_layer_visualization(result, "encoder:block00")

        self.assertEqual(payload["clean_image"].shape, (8, 8, 3))
        self.assertEqual(payload["adversarial_image"].shape, (8, 8, 3))
        self.assertEqual(payload["diff_image"].shape, (8, 8, 3))
        self.assertGreaterEqual(payload["mean_abs_diff"], 0.0)
        self.assertGreaterEqual(payload["max_abs_diff"], 0.0)

    def test_build_sample_delta_visualization_returns_heatmap_and_stats(self) -> None:
        clean_image = torch.zeros(8, 8, 3, dtype=torch.uint8).numpy()
        adversarial_image = torch.zeros(8, 8, 3, dtype=torch.uint8).numpy()
        adversarial_image[2:4, 2:4, :] = 32
        result = FeaturePreviewResult(
            sample_id="sample_0002",
            attack_name="fgsm",
            epsilon=4.0 / 255.0,
            step_size=4.0 / 255.0,
            steps=1,
            clean_image=clean_image,
            adversarial_image=adversarial_image,
            perturbation_image=torch.zeros(8, 8, 3, dtype=torch.uint8).numpy(),
            sample_delta_heatmap=torch.zeros(8, 8, 3, dtype=torch.uint8).numpy(),
            sample_delta_mean=0.0,
            sample_delta_max=0.0,
            ground_truth=torch.zeros(8, 8, dtype=torch.int64).numpy(),
            clean_prediction=torch.zeros(8, 8, dtype=torch.int64).numpy(),
            adversarial_prediction=torch.zeros(8, 8, dtype=torch.int64).numpy(),
            clean_features={"encoder:block00": torch.randn(1, 4, 4, 4)},
            adversarial_features={"encoder:block00": torch.randn(1, 4, 4, 4)},
            layer_names=["encoder:block00"],
            attack_metadata={},
        )

        payload = build_sample_delta_visualization(result)

        self.assertEqual(payload["delta_image"].shape, (8, 8, 3))
        self.assertGreater(payload["mean_abs_delta"], 0.0)
        self.assertGreater(payload["max_abs_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
