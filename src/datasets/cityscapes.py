from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(slots=True)
class CityscapesSample:
    sample_id: str
    image_path: Path
    mask_path: Path
    relative_image_path: Path
    relative_mask_path: Path


def discover_cityscapes_samples(root: str | Path, split: str = "val") -> list[CityscapesSample]:
    dataset_root = Path(root) / CityscapesSegmentationDataset.base_dir
    image_split_root = dataset_root / CityscapesSegmentationDataset.image_dir / split
    mask_split_root = dataset_root / CityscapesSegmentationDataset.mask_dir / split

    if not image_split_root.exists():
        raise FileNotFoundError(f"Cityscapes image split directory not found: {image_split_root}")
    if not mask_split_root.exists():
        raise FileNotFoundError(f"Cityscapes mask split directory not found: {mask_split_root}")

    samples: list[CityscapesSample] = []
    for image_path in sorted(image_split_root.rglob("*_leftImg8bit.png")):
        relative_image_path = image_path.relative_to(image_split_root)
        sample_stem = image_path.name.removesuffix("_leftImg8bit.png")
        relative_mask_path = relative_image_path.parent / f"{sample_stem}_gtFine_labelIds.png"
        mask_path = mask_split_root / relative_mask_path
        if not mask_path.exists():
            continue
        samples.append(
            CityscapesSample(
                sample_id=relative_image_path.with_suffix("").as_posix(),
                image_path=image_path,
                mask_path=mask_path,
                relative_image_path=relative_image_path,
                relative_mask_path=relative_mask_path,
            )
        )

    if not samples:
        raise ValueError(f"No Cityscapes samples found under split directory: {image_split_root}")
    return samples


class CityscapesSegmentationDataset(Dataset):
    """Reference-compatible Cityscapes split loader used by local preview and evaluation helpers."""

    base_dir = Path("cityscapes")
    image_dir = Path("leftImg8bit")
    mask_dir = Path("gtFine")
    num_classes = 19

    def __init__(
        self,
        root: str | Path,
        split: str = "val",
        resize_short: int = 512,
        crop_size: int = 512,
        remap_ignore_to_background: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.resize_short = int(resize_short)
        self.crop_size = int(crop_size)
        self.remap_ignore_to_background = remap_ignore_to_background
        self.samples = discover_cityscapes_samples(self.root, split=self.split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[index]

        with Image.open(sample.image_path) as image:
            rgb_image = image.convert("RGB")
            rgb_image = self._resize_short_and_center_crop(rgb_image, is_mask=False)

        with Image.open(sample.mask_path) as mask:
            mask_image = self._resize_short_and_center_crop(mask, is_mask=True)

        image_tensor = self._to_tensor(rgb_image)
        mask_tensor = self._mask_to_tensor(mask_image)
        return image_tensor, mask_tensor, sample.relative_image_path.as_posix()

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
