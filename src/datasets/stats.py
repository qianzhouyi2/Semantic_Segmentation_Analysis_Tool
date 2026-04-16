from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.datasets.scanner import DatasetPair
from src.io.image_io import load_mask


@dataclass(slots=True)
class ClassRow:
    class_id: int
    class_name: str
    pixel_count: int
    pixel_ratio: float
    sample_count: int
    sample_ratio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "pixel_count": self.pixel_count,
            "pixel_ratio": self.pixel_ratio,
            "sample_count": self.sample_count,
            "sample_ratio": self.sample_ratio,
        }


@dataclass(slots=True)
class DatasetStatistics:
    total_samples: int
    total_labeled_pixels: int
    class_rows: list[ClassRow]
    background_pixel_ratio: float
    foreground_pixel_ratio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "total_samples": self.total_samples,
            "total_labeled_pixels": self.total_labeled_pixels,
            "background_pixel_ratio": self.background_pixel_ratio,
            "foreground_pixel_ratio": self.foreground_pixel_ratio,
            "class_rows": [row.to_dict() for row in self.class_rows],
        }


def compute_class_statistics(
    pairs: list[DatasetPair],
    class_names: dict[int, str] | None = None,
    ignore_index: int | None = None,
    background_ids: tuple[int, ...] = (0,),
) -> DatasetStatistics:
    class_names = class_names or {}

    pixel_counter: Counter[int] = Counter()
    sample_counter: Counter[int] = Counter()
    total_labeled_pixels = 0
    background_pixels = 0

    for pair in pairs:
        mask = load_mask(pair.mask_path)
        flattened = mask.reshape(-1)
        if ignore_index is not None:
            flattened = flattened[flattened != ignore_index]

        if flattened.size == 0:
            continue

        unique_labels, counts = np.unique(flattened, return_counts=True)
        total_labeled_pixels += int(counts.sum())
        for raw_label, raw_count in zip(unique_labels.tolist(), counts.tolist(), strict=True):
            label = int(raw_label)
            count = int(raw_count)
            pixel_counter[label] += count
            sample_counter[label] += 1
            if label in background_ids:
                background_pixels += count

    all_class_ids = sorted(set(pixel_counter) | set(class_names))
    total_samples = len(pairs)
    class_rows: list[ClassRow] = []
    for class_id in all_class_ids:
        pixel_count = pixel_counter[class_id]
        sample_count = sample_counter[class_id]
        class_rows.append(
            ClassRow(
                class_id=class_id,
                class_name=class_names.get(class_id, f"class_{class_id}"),
                pixel_count=pixel_count,
                pixel_ratio=(pixel_count / total_labeled_pixels) if total_labeled_pixels else 0.0,
                sample_count=sample_count,
                sample_ratio=(sample_count / total_samples) if total_samples else 0.0,
            )
        )

    background_ratio = (background_pixels / total_labeled_pixels) if total_labeled_pixels else 0.0
    return DatasetStatistics(
        total_samples=total_samples,
        total_labeled_pixels=total_labeled_pixels,
        class_rows=class_rows,
        background_pixel_ratio=background_ratio,
        foreground_pixel_ratio=1.0 - background_ratio if total_labeled_pixels else 0.0,
    )
