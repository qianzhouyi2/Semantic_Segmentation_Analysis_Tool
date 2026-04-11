from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from src.io.image_io import load_image_rgb, load_mask


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _default_color(class_id: int) -> tuple[int, int, int]:
    return (
        (37 * class_id + 53) % 255,
        (17 * class_id + 109) % 255,
        (67 * class_id + 29) % 255,
    )


def colorize_mask(mask: np.ndarray, palette: dict[int, tuple[int, int, int]] | None = None) -> np.ndarray:
    colors = np.zeros((*mask.shape, 3), dtype=np.uint8)
    palette = palette or {}
    for class_id in np.unique(mask):
        label = int(class_id)
        colors[mask == label] = palette.get(label, _default_color(label))
    return colors


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    palette: dict[int, tuple[int, int, int]] | None = None,
    alpha: float = 0.45,
    ignore_index: int | None = None,
) -> np.ndarray:
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask must share the same spatial size.")

    overlay = colorize_mask(mask, palette).astype(np.float32)
    base = image.astype(np.float32).copy()
    valid = np.ones(mask.shape, dtype=bool)
    if ignore_index is not None:
        valid &= mask != ignore_index

    base[valid] = (1.0 - alpha) * base[valid] + alpha * overlay[valid]
    return np.clip(base, 0, 255).astype(np.uint8)


def render_triplet(
    image: np.ndarray,
    ground_truth: np.ndarray | None = None,
    prediction: np.ndarray | None = None,
    palette: dict[int, tuple[int, int, int]] | None = None,
    alpha: float = 0.45,
    ignore_index: int | None = None,
):
    panels = [
        ("Image", image),
        ("Ground Truth", overlay_mask(image, ground_truth, palette, alpha, ignore_index) if ground_truth is not None else image),
        ("Prediction", overlay_mask(image, prediction, palette, alpha, ignore_index) if prediction is not None else image),
    ]

    figure, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
    for axis, (title, panel) in zip(axes, panels, strict=True):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    return figure


def render_triplet_from_paths(
    image_path: str | Path,
    ground_truth_path: str | Path | None = None,
    prediction_path: str | Path | None = None,
    palette: dict[int, tuple[int, int, int]] | None = None,
    alpha: float = 0.45,
    ignore_index: int | None = None,
):
    image = load_image_rgb(image_path)
    ground_truth = load_mask(ground_truth_path) if ground_truth_path is not None else None
    prediction = load_mask(prediction_path) if prediction_path is not None else None
    return render_triplet(image, ground_truth, prediction, palette, alpha, ignore_index)
