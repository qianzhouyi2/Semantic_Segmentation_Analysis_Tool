from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PASCAL_VOC_CLASS_NAMES: dict[int, str] = {
    0: "background",
    1: "airplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorcycle",
    15: "person",
    16: "potted-plant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tv",
}


@dataclass(slots=True)
class PascalVOCSample:
    sample_id: str
    image_path: Path
    mask_path: Path


def discover_pascal_voc_samples(root: str | Path, split: str = "val") -> list[PascalVOCSample]:
    dataset_root = Path(root) / PascalVOCValidationDataset.base_dir
    split_file = dataset_root / PascalVOCValidationDataset.split_dir / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Pascal VOC split file not found: {split_file}")

    sample_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not sample_ids:
        raise ValueError(f"No samples found in split file: {split_file}")

    samples: list[PascalVOCSample] = []
    for sample_id in sample_ids:
        image_path = dataset_root / PascalVOCValidationDataset.image_dir / f"{sample_id}.jpg"
        mask_path = dataset_root / PascalVOCValidationDataset.mask_dir / f"{sample_id}.png"
        if not image_path.exists() or not mask_path.exists():
            continue
        samples.append(PascalVOCSample(sample_id=sample_id, image_path=image_path, mask_path=mask_path))
    return samples


class PascalVOCValidationDataset(Dataset):
    """Reference-compatible Pascal VOC split loader used by local evaluation."""

    base_dir = Path("VOCdevkit") / "VOC2012"
    split_dir = Path("ImageSets") / "Segmentation"
    image_dir = Path("JPEGImages")
    mask_dir = Path("SegmentationClass")
    num_classes = 21

    def __init__(
        self,
        root: str | Path,
        split: str = "val",
        resize_short: int = 473,
        crop_size: int = 473,
        remap_ignore_to_background: bool = True,
    ) -> None:
        self.root = Path(root)
        self.dataset_root = self.root / self.base_dir
        self.split = str(split)
        self.resize_short = int(resize_short)
        self.crop_size = int(crop_size)
        self.remap_ignore_to_background = remap_ignore_to_background

        split_file = self.dataset_root / self.split_dir / f"{self.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Pascal VOC split file not found: {split_file}")

        self.sample_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.sample_ids:
            raise ValueError(f"No samples found in split file: {split_file}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample_id = self.sample_ids[index]
        image_path = self.dataset_root / self.image_dir / f"{sample_id}.jpg"
        mask_path = self.dataset_root / self.mask_dir / f"{sample_id}.png"

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            rgb_image = self._resize_short_and_center_crop(rgb_image, is_mask=False)

        with Image.open(mask_path) as mask:
            mask_image = self._resize_short_and_center_crop(mask, is_mask=True)

        image_tensor = self._to_tensor(rgb_image)
        mask_tensor = self._mask_to_tensor(mask_image)
        return image_tensor, mask_tensor, f"{sample_id}.jpg"

    def _resize_short_and_center_crop(self, image: Image.Image, is_mask: bool) -> Image.Image:
        width, height = image.size
        if width > height:
            out_height = self.resize_short
            out_width = int(1.0 * width * out_height / height)
        else:
            out_width = self.resize_short
            out_height = int(1.0 * height * out_width / width)

        interpolation = Image.NEAREST if is_mask else Image.BILINEAR
        image = image.resize((out_width, out_height), interpolation)

        left = int(round((out_width - self.crop_size) / 2.0))
        top = int(round((out_height - self.crop_size) / 2.0))
        return image.crop((left, top, left + self.crop_size, top + self.crop_size))

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1))

    def _mask_to_tensor(self, mask: Image.Image) -> torch.Tensor:
        array = np.asarray(mask, dtype=np.int64)
        if self.remap_ignore_to_background:
            array[array == 255] = 0
        return torch.from_numpy(array)
