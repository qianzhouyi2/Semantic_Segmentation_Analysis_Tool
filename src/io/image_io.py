from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_MASK_SUFFIXES = (".png", ".bmp", ".tif", ".tiff")


def load_image_rgb(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"))


def load_mask(path: str | Path) -> np.ndarray:
    mask_path = Path(path)
    with Image.open(mask_path) as image:
        mask = np.asarray(image)

    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int64, copy=False)


def get_image_size(path: str | Path) -> tuple[int, int]:
    image_path = Path(path)
    with Image.open(image_path) as image:
        return image.size


def save_image(path: str | Path, array: np.ndarray) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = array
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)

    Image.fromarray(data).save(output_path)
    return output_path
