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

from src.datasets.ade20k import ADE20KSegmentationDataset, discover_ade20k_samples


class ADE20KDatasetTest(unittest.TestCase):
    def test_discover_ade20k_samples_reads_validation_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "ADEChallengeData2016" / "images" / "validation"
            mask_dir = root / "ADEChallengeData2016" / "annotations" / "validation"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / "ADE_val_00000001.jpg")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(mask_dir / "ADE_val_00000001.png")
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / "ADE_val_00000002.jpg")

            samples = discover_ade20k_samples(root, split="validation")

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, "ADE_val_00000001")
            self.assertEqual(samples[0].relative_mask_path, Path("ADE_val_00000001.png"))

    def test_ade20k_dataset_accepts_val_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "ADEChallengeData2016" / "images" / "validation"
            mask_dir = root / "ADEChallengeData2016" / "annotations" / "validation"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(image_dir / "ADE_val_00000001.jpg")
            mask = np.zeros((12, 16), dtype=np.uint8)
            mask[0, 0] = 255
            Image.fromarray(mask).save(mask_dir / "ADE_val_00000001.png")

            dataset = ADE20KSegmentationDataset(root, split="val", resize_short=16, crop_size=16)
            image, target, filename = dataset[0]

            self.assertEqual(tuple(image.shape), (3, 16, 16))
            self.assertEqual(tuple(target.shape), (16, 16))
            self.assertEqual(filename, "ADE_val_00000001.jpg")
            self.assertEqual(int(target[0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
