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

from src.datasets.scanner import scan_dataset


class DatasetScannerTest(unittest.TestCase):
    def test_scan_dataset_flags_empty_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "images"
            mask_dir = root / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()

            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / "sample.png")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(mask_dir / "sample.png")

            result = scan_dataset(image_dir=image_dir, mask_dir=mask_dir)
            self.assertEqual(len(result.matched_pairs), 1)
            self.assertEqual(result.empty_masks, ["sample"])


if __name__ == "__main__":
    unittest.main()
