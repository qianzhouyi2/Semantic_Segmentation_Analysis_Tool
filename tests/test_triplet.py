from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.triplet import discover_triplet_samples, overlay_mask, render_triplet_from_paths


class TripletVisualizationTest(unittest.TestCase):
    def test_discover_triplet_samples_matches_prediction_by_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "images"
            mask_dir = root / "masks"
            pred_dir = root / "preds"
            image_dir.mkdir()
            mask_dir.mkdir()
            pred_dir.mkdir()

            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / "sample_a.png")
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / "sample_b.png")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(mask_dir / "sample_a.png")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(mask_dir / "sample_b.png")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(pred_dir / "sample_a.png")

            samples = discover_triplet_samples(
                image_dir=image_dir,
                ground_truth_dir=mask_dir,
                prediction_dir=pred_dir,
            )

            self.assertEqual([sample.key for sample in samples], ["sample_a", "sample_b"])
            self.assertEqual(samples[0].prediction_path, pred_dir / "sample_a.png")
            self.assertIsNone(samples[1].prediction_path)

    def test_overlay_mask_respects_ignore_index(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.array([[1, 255], [1, 0]], dtype=np.int64)

        overlay = overlay_mask(
            image=image,
            mask=mask,
            palette={0: (0, 0, 0), 1: (10, 20, 30), 255: (250, 250, 250)},
            alpha=1.0,
            ignore_index=255,
        )

        self.assertTrue(np.array_equal(overlay[0, 0], np.array([10, 20, 30], dtype=np.uint8)))
        self.assertTrue(np.array_equal(overlay[0, 1], np.array([0, 0, 0], dtype=np.uint8)))

    def test_render_triplet_from_paths_returns_three_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.png"
            gt_path = root / "gt.png"
            pred_path = root / "pred.png"

            Image.fromarray(np.full((6, 6, 3), 64, dtype=np.uint8)).save(image_path)
            Image.fromarray(np.zeros((6, 6), dtype=np.uint8)).save(gt_path)
            Image.fromarray(np.ones((6, 6), dtype=np.uint8)).save(pred_path)

            figure = render_triplet_from_paths(
                image_path=image_path,
                ground_truth_path=gt_path,
                prediction_path=pred_path,
                palette={0: (0, 0, 0), 1: (0, 255, 0)},
                class_names={0: "background", 1: "foreground"},
                show_legend=True,
            )

            self.assertEqual(len(figure.axes), 3)
            self.assertEqual([axis.get_title() for axis in figure.axes], ["Image", "Ground Truth", "Prediction"])
            figure.clf()


if __name__ == "__main__":
    unittest.main()
