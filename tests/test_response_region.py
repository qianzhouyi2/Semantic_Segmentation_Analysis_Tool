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
from src.visualization.response_region import (
    build_response_region_visualization,
    compute_input_response_heatmap,
)


class ResponseRegionVisualizationTest(unittest.TestCase):
    def _build_test_adapter(self) -> TorchSegmentationModelAdapter:
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(4, 2, kernel_size=1, bias=True),
        )
        return TorchSegmentationModelAdapter(model=model, num_classes=2, device="cpu")

    def test_compute_input_response_heatmap_returns_normalized_heatmap(self) -> None:
        adapter = self._build_test_adapter()
        image = torch.linspace(0.0, 1.0, steps=16, dtype=torch.float32).reshape(1, 4, 4)

        heatmap, metadata = compute_input_response_heatmap(adapter, image, class_id=0)

        self.assertEqual(heatmap.shape, (4, 4))
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)
        self.assertIn("target_pixels", metadata)
        self.assertIn("score", metadata)
        self.assertIn("used_fallback", metadata)

    def test_build_response_region_visualization_returns_displayable_images(self) -> None:
        adapter = self._build_test_adapter()
        clean_tensor = torch.linspace(0.0, 1.0, steps=16, dtype=torch.float32).reshape(1, 4, 4)
        adversarial_tensor = (clean_tensor.flip(-1) * 0.85).contiguous()
        clean_image = torch.full((4, 4, 3), 128, dtype=torch.uint8).numpy()
        adversarial_image = torch.full((4, 4, 3), 112, dtype=torch.uint8).numpy()

        payload = build_response_region_visualization(
            model=adapter,
            clean_tensor=clean_tensor,
            adversarial_tensor=adversarial_tensor,
            clean_image=clean_image,
            adversarial_image=adversarial_image,
            class_id=0,
            threshold_percentile=80,
        )

        self.assertEqual(payload.clean_overlay.shape, (4, 4, 3))
        self.assertEqual(payload.adversarial_overlay.shape, (4, 4, 3))
        self.assertEqual(payload.diff_overlay.shape, (4, 4, 3))
        self.assertEqual(payload.clean_region_overlay.shape, (4, 4, 3))
        self.assertEqual(payload.adversarial_region_overlay.shape, (4, 4, 3))
        self.assertEqual(payload.overlap_region_overlay.shape, (4, 4, 3))
        self.assertEqual(payload.clean_heatmap.shape, (4, 4))
        self.assertEqual(payload.clean_region_mask.shape, (4, 4))
        self.assertEqual(payload.clean_region_mask.dtype, bool)
        self.assertGreaterEqual(payload.clean_active_ratio, 0.0)
        self.assertLessEqual(payload.clean_active_ratio, 1.0)
        self.assertGreaterEqual(payload.adversarial_active_ratio, 0.0)
        self.assertLessEqual(payload.adversarial_active_ratio, 1.0)
        self.assertGreaterEqual(payload.overlap_iou, 0.0)
        self.assertLessEqual(payload.overlap_iou, 1.0)
        self.assertIsInstance(payload.clean_used_fallback, bool)


if __name__ == "__main__":
    unittest.main()
