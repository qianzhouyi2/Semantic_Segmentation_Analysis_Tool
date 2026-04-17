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

from src.datasets.voc import PascalVOCValidationDataset, discover_pascal_voc_samples


class PascalVOCTest(unittest.TestCase):
    def test_discover_pascal_voc_samples_reads_split_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            voc_root = root / "VOCdevkit" / "VOC2012"
            image_dir = voc_root / "JPEGImages"
            mask_dir = voc_root / "SegmentationClass"
            split_dir = voc_root / "ImageSets" / "Segmentation"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            split_dir.mkdir(parents=True)

            (split_dir / "val.txt").write_text("2007_000001\n2007_000002\n", encoding="utf-8")
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / "2007_000001.jpg")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(mask_dir / "2007_000001.png")
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / "2007_000002.jpg")

            samples = discover_pascal_voc_samples(root, split="val")

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, "2007_000001")
            self.assertEqual(samples[0].image_path, image_dir / "2007_000001.jpg")
            self.assertEqual(samples[0].mask_path, mask_dir / "2007_000001.png")

    def test_discover_pascal_voc_samples_raises_for_missing_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                discover_pascal_voc_samples(tmp_dir, split="val")

    def test_pascal_voc_dataset_accepts_non_val_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            voc_root = root / "VOCdevkit" / "VOC2012"
            image_dir = voc_root / "JPEGImages"
            mask_dir = voc_root / "SegmentationClass"
            split_dir = voc_root / "ImageSets" / "Segmentation"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            split_dir.mkdir(parents=True)

            (split_dir / "train.txt").write_text("2007_000010\n", encoding="utf-8")
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / "2007_000010.jpg")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(mask_dir / "2007_000010.png")

            dataset = PascalVOCValidationDataset(root, split="train", resize_short=8, crop_size=8)

            self.assertEqual(len(dataset), 1)
            image, mask, filename = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 8, 8))
            self.assertEqual(tuple(mask.shape), (8, 8))
            self.assertEqual(filename, "2007_000010.jpg")

    def test_pascal_voc_dataset_can_preserve_ignore_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            voc_root = root / "VOCdevkit" / "VOC2012"
            image_dir = voc_root / "JPEGImages"
            mask_dir = voc_root / "SegmentationClass"
            split_dir = voc_root / "ImageSets" / "Segmentation"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            split_dir.mkdir(parents=True)

            (split_dir / "val.txt").write_text("2007_000011\n", encoding="utf-8")
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / "2007_000011.jpg")
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[0, 0] = 255
            Image.fromarray(mask).save(mask_dir / "2007_000011.png")

            preserved_dataset = PascalVOCValidationDataset(
                root,
                split="val",
                resize_short=8,
                crop_size=8,
                remap_ignore_to_background=False,
            )
            remapped_dataset = PascalVOCValidationDataset(
                root,
                split="val",
                resize_short=8,
                crop_size=8,
                remap_ignore_to_background=True,
            )

            _, preserved_mask, _ = preserved_dataset[0]
            _, remapped_mask, _ = remapped_dataset[0]

            self.assertEqual(int(preserved_mask[0, 0].item()), 255)
            self.assertEqual(int(remapped_mask[0, 0].item()), 0)


if __name__ == "__main__":
    unittest.main()
