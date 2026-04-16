from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.io.image_io import DEFAULT_IMAGE_SUFFIXES, DEFAULT_MASK_SUFFIXES, get_image_size, load_mask


def _normalize_suffixes(values: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        suffix = value if value.startswith(".") else f".{value}"
        normalized.append(suffix.lower())
    return tuple(normalized)


def _path_key(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix()


def discover_files(directory: str | Path, suffixes: Iterable[str]) -> dict[str, Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    suffix_set = set(_normalize_suffixes(suffixes))
    discovered: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffix_set:
            discovered[_path_key(path, root)] = path
    return discovered


@dataclass(slots=True)
class DatasetPair:
    key: str
    image_path: Path
    mask_path: Path


@dataclass(slots=True)
class InspectedSample:
    key: str
    image_size: tuple[int, int]
    mask_size: tuple[int, int]
    unique_labels: list[int]
    invalid_labels: list[int]
    is_empty_mask: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "image_size": list(self.image_size),
            "mask_size": list(self.mask_size),
            "unique_labels": self.unique_labels,
            "invalid_labels": self.invalid_labels,
            "is_empty_mask": self.is_empty_mask,
        }


@dataclass(slots=True)
class DatasetScanResult:
    image_dir: Path
    mask_dir: Path
    total_images: int
    total_masks: int
    matched_pairs: list[DatasetPair]
    missing_masks: list[str]
    orphan_masks: list[str]
    mismatched_shapes: list[str]
    empty_masks: list[str]
    invalid_label_samples: list[str]
    inspected_samples: list[InspectedSample]

    def to_dict(self) -> dict[str, object]:
        return {
            "image_dir": str(self.image_dir),
            "mask_dir": str(self.mask_dir),
            "total_images": self.total_images,
            "total_masks": self.total_masks,
            "matched_pairs": [
                {
                    "key": pair.key,
                    "image_path": str(pair.image_path),
                    "mask_path": str(pair.mask_path),
                }
                for pair in self.matched_pairs
            ],
            "missing_masks": self.missing_masks,
            "orphan_masks": self.orphan_masks,
            "mismatched_shapes": self.mismatched_shapes,
            "empty_masks": self.empty_masks,
            "invalid_label_samples": self.invalid_label_samples,
            "inspected_samples": [sample.to_dict() for sample in self.inspected_samples],
        }


def scan_dataset(
    image_dir: str | Path,
    mask_dir: str | Path,
    image_suffixes: Iterable[str] = DEFAULT_IMAGE_SUFFIXES,
    mask_suffixes: Iterable[str] = DEFAULT_MASK_SUFFIXES,
    allowed_label_ids: set[int] | None = None,
    ignore_index: int | None = None,
    background_ids: tuple[int, ...] = (0,),
) -> DatasetScanResult:
    image_root = Path(image_dir)
    mask_root = Path(mask_dir)
    image_files = discover_files(image_root, image_suffixes)
    mask_files = discover_files(mask_root, mask_suffixes)

    image_keys = set(image_files)
    mask_keys = set(mask_files)
    matched_keys = sorted(image_keys & mask_keys)

    matched_pairs = [
        DatasetPair(key=key, image_path=image_files[key], mask_path=mask_files[key]) for key in matched_keys
    ]
    missing_masks = sorted(image_keys - mask_keys)
    orphan_masks = sorted(mask_keys - image_keys)

    mismatched_shapes: list[str] = []
    empty_masks: list[str] = []
    invalid_label_samples: list[str] = []
    inspected_samples: list[InspectedSample] = []

    for pair in matched_pairs:
        image_size = get_image_size(pair.image_path)
        mask = load_mask(pair.mask_path)
        mask_size = (int(mask.shape[1]), int(mask.shape[0]))
        unique_labels = sorted(int(item) for item in np.unique(mask).tolist())

        invalid_labels = []
        if allowed_label_ids is not None:
            invalid_labels = [
                label
                for label in unique_labels
                if label not in allowed_label_ids and (ignore_index is None or label != ignore_index)
            ]

        valid_foreground = [
            label
            for label in unique_labels
            if label not in background_ids and (ignore_index is None or label != ignore_index)
        ]
        is_empty_mask = len(valid_foreground) == 0

        if image_size != mask_size:
            mismatched_shapes.append(pair.key)
        if is_empty_mask:
            empty_masks.append(pair.key)
        if invalid_labels:
            invalid_label_samples.append(pair.key)

        inspected_samples.append(
            InspectedSample(
                key=pair.key,
                image_size=image_size,
                mask_size=mask_size,
                unique_labels=unique_labels,
                invalid_labels=invalid_labels,
                is_empty_mask=is_empty_mask,
            )
        )

    return DatasetScanResult(
        image_dir=image_root,
        mask_dir=mask_root,
        total_images=len(image_files),
        total_masks=len(mask_files),
        matched_pairs=matched_pairs,
        missing_masks=missing_masks,
        orphan_masks=orphan_masks,
        mismatched_shapes=mismatched_shapes,
        empty_masks=empty_masks,
        invalid_label_samples=invalid_label_samples,
        inspected_samples=inspected_samples,
    )
