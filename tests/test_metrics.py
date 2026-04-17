from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.segmentation import (
    compute_confusion_matrix,
    compute_per_sample_segmentation_metrics,
    summarize_confusion_matrix,
)


class SegmentationMetricsTest(unittest.TestCase):
    def test_perfect_prediction(self) -> None:
        target = np.array([[0, 1], [1, 0]], dtype=np.int64)
        prediction = np.array([[0, 1], [1, 0]], dtype=np.int64)

        confusion = compute_confusion_matrix(target, prediction, num_classes=2)
        metrics = summarize_confusion_matrix(confusion)

        self.assertAlmostEqual(metrics.pixel_accuracy, 1.0)
        self.assertAlmostEqual(metrics.mean_iou, 1.0)
        self.assertAlmostEqual(metrics.mean_dice, 1.0)

    def test_ignore_index(self) -> None:
        target = np.array([[0, 255], [1, 0]], dtype=np.int64)
        prediction = np.array([[0, 1], [1, 1]], dtype=np.int64)

        confusion = compute_confusion_matrix(target, prediction, num_classes=2, ignore_index=255)
        self.assertEqual(confusion.tolist(), [[1, 1], [0, 1]])

    def test_per_sample_metrics_ignore_zero_union_classes(self) -> None:
        target = np.array([[0, 1], [1, 1]], dtype=np.int64)
        prediction = np.array([[0, 0], [1, 1]], dtype=np.int64)

        metrics = compute_per_sample_segmentation_metrics(target, prediction, num_classes=3)

        self.assertAlmostEqual(metrics.pixel_accuracy, 0.75)
        self.assertAlmostEqual(metrics.sample_miou, (0.5 + (2.0 / 3.0)) / 2.0)
        self.assertAlmostEqual(metrics.sample_dice, ((2.0 / 3.0) + 0.8) / 2.0)
        self.assertEqual(metrics.valid_class_count, 2)


if __name__ == "__main__":
    unittest.main()
