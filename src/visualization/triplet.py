from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from src.datasets.scanner import discover_files
from src.io.image_io import DEFAULT_IMAGE_SUFFIXES, DEFAULT_MASK_SUFFIXES
from src.io.image_io import load_image_rgb, load_mask


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


@dataclass(slots=True)
class TripletSample:
    key: str
    image_path: Path
    ground_truth_path: Path | None
    prediction_path: Path | None


def discover_triplet_samples(
    image_dir: str | Path,
    ground_truth_dir: str | Path | None = None,
    prediction_dir: str | Path | None = None,
    image_suffixes: tuple[str, ...] = DEFAULT_IMAGE_SUFFIXES,
    mask_suffixes: tuple[str, ...] = DEFAULT_MASK_SUFFIXES,
    prediction_suffixes: tuple[str, ...] | None = None,
    require_prediction: bool = False,
) -> list[TripletSample]:
    image_files = discover_files(image_dir, image_suffixes)
    ground_truth_files = discover_files(ground_truth_dir, mask_suffixes) if ground_truth_dir is not None else {}
    prediction_files = (
        discover_files(prediction_dir, prediction_suffixes or mask_suffixes) if prediction_dir is not None else {}
    )

    candidate_keys = set(image_files)
    if ground_truth_dir is not None:
        candidate_keys &= set(ground_truth_files)
    if require_prediction:
        candidate_keys &= set(prediction_files)

    return [
        TripletSample(
            key=key,
            image_path=image_files[key],
            ground_truth_path=ground_truth_files.get(key),
            prediction_path=prediction_files.get(key),
        )
        for key in sorted(candidate_keys)
    ]


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
    class_names: dict[int, str] | None = None,
    alpha: float = 0.45,
    ignore_index: int | None = None,
    show_legend: bool = False,
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

    if show_legend:
        class_names = class_names or {}
        palette = palette or {}
        labels = set()
        if ground_truth is not None:
            labels.update(int(label) for label in np.unique(ground_truth).tolist())
        if prediction is not None:
            labels.update(int(label) for label in np.unique(prediction).tolist())
        if ignore_index is not None:
            labels.discard(ignore_index)

        handles = [
            Patch(
                facecolor=np.asarray(palette.get(label, _default_color(label)), dtype=np.float32) / 255.0,
                edgecolor="black",
                label=class_names.get(label, f"class_{label}"),
            )
            for label in sorted(labels)
        ]
        if handles:
            figure.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False)
            figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
            return figure

    figure.tight_layout()
    return figure


def render_triplet_from_paths(
    image_path: str | Path,
    ground_truth_path: str | Path | None = None,
    prediction_path: str | Path | None = None,
    palette: dict[int, tuple[int, int, int]] | None = None,
    class_names: dict[int, str] | None = None,
    alpha: float = 0.45,
    ignore_index: int | None = None,
    show_legend: bool = False,
):
    image = load_image_rgb(image_path)
    ground_truth = load_mask(ground_truth_path) if ground_truth_path is not None else None
    prediction = load_mask(prediction_path) if prediction_path is not None else None
    return render_triplet(image, ground_truth, prediction, palette, class_names, alpha, ignore_index, show_legend)
