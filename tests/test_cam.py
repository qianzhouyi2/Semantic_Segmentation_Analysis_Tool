from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architectures.segmenter import create_segmenter
from src.models.base import TorchSegmentationModelAdapter
from src.visualization.cam import (
    build_cam_visualization,
    compute_feature_grad_cam,
    discover_cam_supported_feature_keys,
    select_default_cam_feature_key,
    select_representative_cam_feature_keys,
)


class CamVisualizationTest(unittest.TestCase):
    def test_compute_feature_grad_cam_returns_normalized_heatmap(self) -> None:
        model = nn.Conv2d(1, 2, kernel_size=1, bias=True)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[[[2.0]]], [[[-2.0]]]]))
            model.bias.copy_(torch.tensor([0.5, -0.5]))
        adapter = TorchSegmentationModelAdapter(model=model, num_classes=2, device="cpu")
        image = torch.full((1, 4, 4), 0.75, dtype=torch.float32)

        heatmap, metadata = compute_feature_grad_cam(adapter, image, feature_key="logits", class_id=0)

        self.assertEqual(heatmap.shape, (4, 4))
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)
        self.assertIn("target_pixels", metadata)
        self.assertIn("used_fallback", metadata)

    def test_compute_feature_grad_cam_supports_segmenter_encoder_blocks(self) -> None:
        segmenter = create_segmenter(
            {
                "image_size": (32, 32),
                "patch_size": 16,
                "n_layers": 2,
                "d_model": 384,
                "n_heads": 6,
                "normalization": "vit",
                "distilled": False,
                "backbone": "vit_small_patch16_224",
                "dropout": 0.0,
                "drop_path_rate": 0.0,
                "decoder": {"name": "linear"},
                "n_cls": 2,
            },
            backbone="vit_small_patch16_224",
        )
        adapter = TorchSegmentationModelAdapter(model=segmenter, num_classes=2, device="cpu")
        image = torch.rand((3, 32, 32), dtype=torch.float32)

        heatmap, metadata = compute_feature_grad_cam(adapter, image, feature_key="encoder:block00", class_id=0)

        self.assertEqual(heatmap.shape, (32, 32))
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)
        self.assertIn("target_pixels", metadata)
        self.assertIn("used_fallback", metadata)

    def test_build_cam_visualization_returns_displayable_images(self) -> None:
        model = nn.Conv2d(1, 2, kernel_size=1, bias=True)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[[[2.0]]], [[[-2.0]]]]))
            model.bias.copy_(torch.tensor([0.5, -0.5]))
        adapter = TorchSegmentationModelAdapter(model=model, num_classes=2, device="cpu")
        clean_tensor = torch.tensor(
            [
                [1.0, 0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7, 0.6],
                [0.4, 0.3, 0.2, 0.1],
                [0.3, 0.2, 0.1, 0.0],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        adversarial_tensor = torch.flip(clean_tensor, dims=(1,))
        clean_image = torch.full((4, 4, 3), 128, dtype=torch.uint8).numpy()
        adversarial_image = torch.full((4, 4, 3), 112, dtype=torch.uint8).numpy()
        ground_truth = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=torch.int64,
        ).numpy()
        clean_prediction = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=torch.int64,
        ).numpy()

        payload = build_cam_visualization(
            model=adapter,
            clean_tensor=clean_tensor,
            adversarial_tensor=adversarial_tensor,
            clean_image=clean_image,
            adversarial_image=adversarial_image,
            feature_key="logits",
            class_id=0,
            ground_truth=ground_truth,
            clean_prediction=clean_prediction,
        )

        self.assertEqual(payload.clean_overlay.shape, (4, 4, 3))
        self.assertEqual(payload.adversarial_overlay.shape, (4, 4, 3))
        self.assertEqual(payload.diff_image.shape, (4, 4, 3))
        self.assertGreaterEqual(payload.diff_mean, 0.0)
        self.assertGreater(payload.clean_top20_area_ratio, 0.0)
        self.assertGreater(payload.adversarial_top20_area_ratio, 0.0)
        self.assertGreater(payload.clean_inside_gt_ratio, payload.adversarial_inside_gt_ratio)
        self.assertGreater(payload.clean_inside_clean_prediction_ratio, payload.adversarial_inside_clean_prediction_ratio)
        self.assertIsNotNone(payload.centroid_shift)
        self.assertGreater(float(payload.centroid_shift or 0.0), 0.0)
        self.assertFalse(payload.clean_used_fallback)

    def test_discover_cam_supported_feature_keys_prefers_backbone_stage_keys_by_name(self) -> None:
        adapter = TorchSegmentationModelAdapter(nn.Conv2d(1, 2, kernel_size=1), num_classes=2, device="cpu")

        keys = discover_cam_supported_feature_keys(adapter, ["backbone:stage0:block00", "logits"])

        self.assertEqual(keys, ["backbone:stage0:block00"])

    def test_discover_cam_supported_feature_keys_falls_back_to_logits_when_only_logits_exist(self) -> None:
        adapter = TorchSegmentationModelAdapter(nn.Conv2d(1, 2, kernel_size=1), num_classes=2, device="cpu")

        keys = discover_cam_supported_feature_keys(adapter, ["logits"])

        self.assertEqual(keys, ["logits"])

    def test_select_default_cam_feature_key_prefers_deepest_supported_feature(self) -> None:
        adapter = TorchSegmentationModelAdapter(nn.Conv2d(1, 2, kernel_size=1), num_classes=2, device="cpu")

        feature_key = select_default_cam_feature_key(adapter, ["encoder:block00", "logits"])

        self.assertEqual(feature_key, "encoder:block00")

    def test_select_representative_cam_feature_keys_picks_shallow_mid_and_deep(self) -> None:
        adapter = TorchSegmentationModelAdapter(nn.Conv2d(1, 2, kernel_size=1), num_classes=2, device="cpu")

        with patch(
            "src.visualization.cam.discover_cam_supported_feature_keys",
            return_value=[
                "backbone:stage0:block00",
                "backbone:stage1:block00",
                "backbone:stage2:block00",
                "backbone:stage3:block00",
                "backbone:stage4:block00",
            ],
        ):
            feature_keys = select_representative_cam_feature_keys(adapter, [])

        self.assertEqual(
            feature_keys,
            [
                "backbone:stage0:block00",
                "backbone:stage2:block00",
                "backbone:stage4:block00",
            ],
        )

    def test_select_representative_cam_feature_keys_returns_all_when_supported_layers_are_few(self) -> None:
        adapter = TorchSegmentationModelAdapter(nn.Conv2d(1, 2, kernel_size=1), num_classes=2, device="cpu")

        with patch(
            "src.visualization.cam.discover_cam_supported_feature_keys",
            return_value=["backbone:stage0:block00", "backbone:stage3:block00"],
        ):
            feature_keys = select_representative_cam_feature_keys(adapter, [])

        self.assertEqual(feature_keys, ["backbone:stage0:block00", "backbone:stage3:block00"])


if __name__ == "__main__":
    unittest.main()
