from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.base import TorchSegmentationModelAdapter
from src.visualization.cam import (
    build_cam_visualization,
    compute_feature_grad_cam,
    discover_cam_supported_feature_keys,
    select_default_cam_feature_key,
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

    def test_build_cam_visualization_returns_displayable_images(self) -> None:
        model = nn.Conv2d(1, 2, kernel_size=1, bias=True)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[[[2.0]]], [[[-2.0]]]]))
            model.bias.copy_(torch.tensor([0.5, -0.5]))
        adapter = TorchSegmentationModelAdapter(model=model, num_classes=2, device="cpu")
        clean_tensor = torch.full((1, 4, 4), 0.75, dtype=torch.float32)
        adversarial_tensor = torch.full((1, 4, 4), 0.65, dtype=torch.float32)
        clean_image = torch.full((4, 4, 3), 128, dtype=torch.uint8).numpy()
        adversarial_image = torch.full((4, 4, 3), 112, dtype=torch.uint8).numpy()

        payload = build_cam_visualization(
            model=adapter,
            clean_tensor=clean_tensor,
            adversarial_tensor=adversarial_tensor,
            clean_image=clean_image,
            adversarial_image=adversarial_image,
            feature_key="logits",
            class_id=0,
        )

        self.assertEqual(payload.clean_overlay.shape, (4, 4, 3))
        self.assertEqual(payload.adversarial_overlay.shape, (4, 4, 3))
        self.assertEqual(payload.diff_image.shape, (4, 4, 3))
        self.assertGreaterEqual(payload.diff_mean, 0.0)

    def test_discover_cam_supported_feature_keys_falls_back_to_logits_for_generic_models(self) -> None:
        adapter = TorchSegmentationModelAdapter(nn.Conv2d(1, 2, kernel_size=1), num_classes=2, device="cpu")

        keys = discover_cam_supported_feature_keys(adapter, ["encoder:block00", "logits"])

        self.assertEqual(keys, ["logits"])

    def test_select_default_cam_feature_key_prefers_deepest_supported_feature(self) -> None:
        adapter = TorchSegmentationModelAdapter(nn.Conv2d(1, 2, kernel_size=1), num_classes=2, device="cpu")

        feature_key = select_default_cam_feature_key(adapter, ["encoder:block00", "logits"])

        self.assertEqual(feature_key, "logits")


if __name__ == "__main__":
    unittest.main()
