from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _normalize_ade20k_split(split: str) -> str:
    if split == "val":
        return "validation"
    if split == "train":
        return "training"
    return split


def _candidate_ade20k_split_names(split: str) -> list[str]:
    normalized = _normalize_ade20k_split(split)
    candidates = [normalized]
    if normalized == "validation":
        candidates.append("val")
    elif normalized == "training":
        candidates.append("train")
    return candidates


def _resolve_ade20k_dataset_root(root: str | Path) -> Path:
    root_path = Path(root)
    base_dir = ADE20KSegmentationDataset.base_dir
    image_dir = ADE20KSegmentationDataset.image_dir
    mask_dir = ADE20KSegmentationDataset.mask_dir

    container_candidates = [
        root_path,
        root_path / "ade",
        root_path / "ADE20K",
        root_path / "ade20k",
    ]
    if root_path.name == base_dir.name:
        container_candidates.insert(0, root_path.parent)

    resolved_candidates: list[Path] = []
    seen: set[Path] = set()
    for container in container_candidates:
        direct_dataset_root = container
        nested_dataset_root = container / base_dir
        for candidate in (direct_dataset_root, nested_dataset_root):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / image_dir).exists() and (candidate / mask_dir).exists():
                resolved_candidates.append(candidate)

    if resolved_candidates:
        return resolved_candidates[0]

    return root_path / base_dir


@dataclass(slots=True)
class ADE20KSample:
    sample_id: str
    image_path: Path
    mask_path: Path
    relative_image_path: Path
    relative_mask_path: Path


def discover_ade20k_samples(root: str | Path, split: str = "validation") -> list[ADE20KSample]:
    dataset_root = _resolve_ade20k_dataset_root(root)
    image_split_root = None
    mask_split_root = None
    checked_pairs: list[tuple[Path, Path]] = []
    for split_name in _candidate_ade20k_split_names(str(split)):
        candidate_image_root = dataset_root / ADE20KSegmentationDataset.image_dir / split_name
        candidate_mask_root = dataset_root / ADE20KSegmentationDataset.mask_dir / split_name
        checked_pairs.append((candidate_image_root, candidate_mask_root))
        if candidate_image_root.exists() and candidate_mask_root.exists():
            image_split_root = candidate_image_root
            mask_split_root = candidate_mask_root
            break

    if image_split_root is None or mask_split_root is None:
        checked_text = ", ".join(str(image_root) for image_root, _ in checked_pairs)
        raise FileNotFoundError(
            "ADE20K image split directory not found. "
            f"Checked: {checked_text}. "
            f"Resolved dataset root: {dataset_root}"
        )

    samples: list[ADE20KSample] = []
    for image_path in sorted(image_split_root.rglob("*.jpg")):
        relative_image_path = image_path.relative_to(image_split_root)
        relative_mask_path = relative_image_path.with_suffix(".png")
        mask_path = mask_split_root / relative_mask_path
        if not mask_path.exists():
            continue
        samples.append(
            ADE20KSample(
                sample_id=relative_image_path.with_suffix("").as_posix(),
                image_path=image_path,
                mask_path=mask_path,
                relative_image_path=relative_image_path,
                relative_mask_path=relative_mask_path,
            )
        )

    if not samples:
        raise ValueError(f"No ADE20K samples found under split directory: {image_split_root}")
    return samples


class ADE20KSegmentationDataset(Dataset):
    """ADE20K split loader used by local preview and evaluation helpers."""

    base_dir = Path("ADEChallengeData2016")
    image_dir = Path("images")
    mask_dir = Path("annotations")
    num_classes = 150

    def __init__(
        self,
        root: str | Path,
        split: str = "validation",
        resize_short: int = 512,
        crop_size: int = 512,
        remap_ignore_to_background: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = _normalize_ade20k_split(str(split))
        self.resize_short = int(resize_short)
        self.crop_size = int(crop_size)
        self.remap_ignore_to_background = remap_ignore_to_background
        self.samples = discover_ade20k_samples(self.root, split=self.split)

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
