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

from src.datasets.cityscapes import CityscapesSegmentationDataset, discover_cityscapes_samples


class CityscapesDatasetTest(unittest.TestCase):
    def test_discover_cityscapes_samples_reads_nested_split_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "cityscapes" / "leftImg8bit" / "val" / "frankfurt"
            mask_dir = root / "cityscapes" / "gtFine" / "val" / "frankfurt"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
                image_dir / "frankfurt_000000_000294_leftImg8bit.png"
            )
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(
                mask_dir / "frankfurt_000000_000294_gtFine_labelIds.png"
            )
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
                image_dir / "frankfurt_000001_000295_leftImg8bit.png"
            )

            samples = discover_cityscapes_samples(root, split="val")

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, "frankfurt/frankfurt_000000_000294_leftImg8bit")
            self.assertEqual(
                samples[0].relative_mask_path,
                Path("frankfurt") / "frankfurt_000000_000294_gtFine_labelIds.png",
            )

    def test_cityscapes_dataset_loads_sample_tensor_and_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "cityscapes" / "leftImg8bit" / "val" / "frankfurt"
            mask_dir = root / "cityscapes" / "gtFine" / "val" / "frankfurt"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            Image.fromarray(np.zeros((16, 12, 3), dtype=np.uint8)).save(
                image_dir / "frankfurt_000000_000294_leftImg8bit.png"
            )
            mask = np.zeros((16, 12), dtype=np.uint8)
            mask[0, 0] = 255
            Image.fromarray(mask).save(mask_dir / "frankfurt_000000_000294_gtFine_labelIds.png")

            dataset = CityscapesSegmentationDataset(root, split="val", resize_short=16, crop_size=16)
            image, target, filename = dataset[0]

            self.assertEqual(tuple(image.shape), (3, 16, 16))
            self.assertEqual(tuple(target.shape), (16, 16))
            self.assertEqual(filename, "frankfurt/frankfurt_000000_000294_leftImg8bit.png")
            self.assertEqual(int(target[0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
