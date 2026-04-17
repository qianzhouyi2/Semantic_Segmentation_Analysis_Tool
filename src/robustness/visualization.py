from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

from src.io.image_io import save_image


matplotlib.use("Agg")
import matplotlib.pyplot as plt


HEATMAP_SCALE_MODE_CHOICES = ("independent", "shared", "fixed")


def normalize_perturbation(perturbation: np.ndarray) -> np.ndarray:
    """Convert a perturbation tensor/image into a displayable 0-255 map."""
    if perturbation.size == 0:
        return perturbation.astype(np.uint8)

    data = perturbation.astype(np.float32)
    min_value = float(data.min())
    max_value = float(data.max())
    if max_value == min_value:
        return np.zeros_like(data, dtype=np.uint8)
    normalized = (data - min_value) / (max_value - min_value)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)


def tensor_to_rgb_image(image: torch.Tensor) -> np.ndarray:
    data = image.detach().float().cpu()
    if data.ndim == 4:
        if data.size(0) != 1:
            raise ValueError("tensor_to_rgb_image expects a single image or a batch of size 1.")
        data = data[0]
    if data.ndim != 3:
        raise ValueError(f"Expected image tensor with shape [C, H, W], got {tuple(data.shape)}.")
    if data.size(0) not in {1, 3}:
        raise ValueError(f"Expected 1 or 3 channels for image visualization, got {data.size(0)}.")

    if data.size(0) == 1:
        data = data.repeat(3, 1, 1)
    return (data.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def summarize_feature_map(
    feature_map: torch.Tensor,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    data = feature_map.detach().float().cpu()
    if data.ndim == 4:
        if data.size(0) != 1:
            raise ValueError("summarize_feature_map expects a single feature map or a batch of size 1.")
        data = data[0]
    if data.ndim == 2:
        summary = data.unsqueeze(0).unsqueeze(0)
    elif data.ndim == 3:
        summary = data.pow(2).mean(dim=0, keepdim=True).sqrt().unsqueeze(0)
    else:
        raise ValueError(f"Expected feature map with shape [C, H, W] or [H, W], got {tuple(data.shape)}.")

    if target_size is not None and tuple(summary.shape[-2:]) != tuple(target_size):
        summary = F.interpolate(summary, size=target_size, mode="bilinear", align_corners=False)

    return summary.squeeze(0).squeeze(0).numpy()


def _resolve_single_heatmap_bounds(
    heatmap: np.ndarray,
    *,
    percentile_clip_upper: float = 100.0,
) -> tuple[float, float]:
    data = np.asarray(heatmap, dtype=np.float32)
    if data.size == 0:
        return 0.0, 1.0

    finite_values = data[np.isfinite(data)]
    if finite_values.size == 0:
        return 0.0, 1.0

    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    clipped_percentile = float(np.clip(percentile_clip_upper, 0.0, 100.0))
    if clipped_percentile < 100.0:
        max_value = float(np.percentile(finite_values, clipped_percentile))
    if max_value <= min_value:
        max_value = min_value + 1e-6
    return min_value, max_value


def resolve_heatmap_display_bounds(
    heatmaps: list[np.ndarray],
    *,
    scale_mode: str = "independent",
    percentile_clip_upper: float = 100.0,
    fixed_range: tuple[float, float] = (0.0, 1.0),
) -> list[tuple[float, float]]:
    if scale_mode not in HEATMAP_SCALE_MODE_CHOICES:
        raise ValueError(f"Unsupported scale_mode `{scale_mode}`. Expected one of {HEATMAP_SCALE_MODE_CHOICES}.")

    if not heatmaps:
        return []

    if scale_mode == "fixed":
        lower, upper = fixed_range
        if upper <= lower:
            raise ValueError(f"Expected fixed_range with upper > lower, got {fixed_range}.")
        return [(float(lower), float(upper)) for _ in heatmaps]

    if scale_mode == "shared":
        finite_chunks = [
            np.asarray(heatmap, dtype=np.float32).reshape(-1)
            for heatmap in heatmaps
            if np.asarray(heatmap).size > 0
        ]
        if not finite_chunks:
            return [(0.0, 1.0) for _ in heatmaps]
        all_values = np.concatenate(finite_chunks, axis=0)
        all_values = all_values[np.isfinite(all_values)]
        if all_values.size == 0:
            return [(0.0, 1.0) for _ in heatmaps]
        min_value = float(all_values.min())
        max_value = float(all_values.max())
        clipped_percentile = float(np.clip(percentile_clip_upper, 0.0, 100.0))
        if clipped_percentile < 100.0:
            max_value = float(np.percentile(all_values, clipped_percentile))
        if max_value <= min_value:
            max_value = min_value + 1e-6
        return [(min_value, max_value) for _ in heatmaps]

    return [
        _resolve_single_heatmap_bounds(heatmap, percentile_clip_upper=percentile_clip_upper)
        for heatmap in heatmaps
    ]


def colorize_heatmap(
    heatmap: np.ndarray,
    cmap_name: str = "magma",
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    if heatmap.ndim != 2:
        raise ValueError(f"Expected heatmap with shape [H, W], got {heatmap.shape}.")
    if heatmap.size == 0:
        return np.zeros((*heatmap.shape, 3), dtype=np.uint8)

    data = heatmap.astype(np.float32, copy=False)
    min_value = float(data.min()) if vmin is None else float(vmin)
    max_value = float(data.max()) if vmax is None else float(vmax)
    if max_value > min_value:
        data = (data - min_value) / (max_value - min_value)
    else:
        data = np.zeros_like(data)
    data = np.clip(data, 0.0, 1.0)

    colored = plt.get_cmap(cmap_name)(data)[..., :3]
    return (colored * 255.0).clip(0, 255).astype(np.uint8)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    if heatmap.ndim != 2:
        raise ValueError(f"Expected heatmap with shape [H, W], got {heatmap.shape}.")
    if heatmap.size == 0:
        return np.zeros_like(heatmap, dtype=np.float32)

    data = heatmap.astype(np.float32, copy=False)
    min_value = float(data.min())
    max_value = float(data.max())
    if max_value <= min_value:
        return np.zeros_like(data, dtype=np.float32)
    return (data - min_value) / (max_value - min_value)


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    cmap_name: str = "jet",
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape [H, W, 3], got {image.shape}.")
    if heatmap.shape != image.shape[:2]:
        raise ValueError(f"Heatmap shape {heatmap.shape} must match image shape {image.shape[:2]}.")

    colored = colorize_heatmap(heatmap, cmap_name=cmap_name, vmin=vmin, vmax=vmax).astype(np.float32)
    base = image.astype(np.float32)
    blended = (1.0 - alpha) * base + alpha * colored
    return np.clip(blended, 0, 255).astype(np.uint8)


def overlay_binary_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape [H, W, 3], got {image.shape}.")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} must match image shape {image.shape[:2]}.")

    mask_array = np.asarray(mask, dtype=bool)
    overlay = image.astype(np.float32).copy()
    tint = np.asarray(color, dtype=np.float32)
    overlay[mask_array] = (1.0 - alpha) * overlay[mask_array] + alpha * tint
    return np.clip(overlay, 0, 255).astype(np.uint8)


def summarize_image_delta(
    clean_image: torch.Tensor | np.ndarray,
    adversarial_image: torch.Tensor | np.ndarray,
) -> np.ndarray:
    def _to_hwc_float_array(image: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            data = image.detach().float().cpu()
            if data.ndim == 4:
                if data.size(0) != 1:
                    raise ValueError("Expected a single image or a batch of size 1.")
                data = data[0]
            if data.ndim != 3:
                raise ValueError(f"Expected tensor image with shape [C, H, W], got {tuple(data.shape)}.")
            if data.size(0) not in {1, 3}:
                raise ValueError(f"Expected 1 or 3 channels, got {data.size(0)}.")
            if data.size(0) == 1:
                data = data.repeat(3, 1, 1)
            return data.permute(1, 2, 0).numpy()

        data = np.asarray(image, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"Expected ndarray image with shape [H, W, C], got {data.shape}.")
        if data.shape[2] not in {1, 3}:
            raise ValueError(f"Expected 1 or 3 channels, got {data.shape[2]}.")
        if data.shape[2] == 1:
            data = np.repeat(data, 3, axis=2)
        if data.max(initial=0.0) > 1.0:
            data = data / 255.0
        return data

    clean = _to_hwc_float_array(clean_image)
    adversarial = _to_hwc_float_array(adversarial_image)
    if clean.shape != adversarial.shape:
        raise ValueError(f"Clean and adversarial images must have identical shape, got {clean.shape} vs {adversarial.shape}.")
    return np.abs(adversarial - clean).mean(axis=2)


def _sanitize_name(name: str) -> str:
    safe_chars = [character if character.isalnum() or character in {"_", "-", "."} else "_" for character in name]
    sanitized = "".join(safe_chars).strip("._")
    return sanitized or "layer"


def save_layerwise_feature_visualizations(
    output_dir: str | Path,
    sample_key: str,
    clean_image: torch.Tensor,
    adversarial_image: torch.Tensor,
    perturbation: torch.Tensor,
    clean_features: dict[str, torch.Tensor],
    adversarial_features: dict[str, torch.Tensor],
    max_layers: int = -1,
    cmap_name: str = "magma",
) -> dict[str, object]:
    sample_dir = Path(output_dir) / _sanitize_name(sample_key)
    sample_dir.mkdir(parents=True, exist_ok=True)

    clean_image_rgb = tensor_to_rgb_image(clean_image)
    adversarial_image_rgb = tensor_to_rgb_image(adversarial_image)
    perturbation_rgb = normalize_perturbation(
        perturbation.detach().float().cpu()[0].permute(1, 2, 0).numpy()
        if perturbation.ndim == 4
        else perturbation.detach().float().cpu().permute(1, 2, 0).numpy()
    )

    save_image(sample_dir / "input.png", clean_image_rgb)
    save_image(sample_dir / "adversarial.png", adversarial_image_rgb)
    save_image(sample_dir / "perturbation.png", perturbation_rgb)

    layer_records: list[dict[str, object]] = []
    target_size = clean_image_rgb.shape[:2]
    shared_layer_names = [name for name in clean_features if name in adversarial_features]
    if max_layers > 0:
        shared_layer_names = shared_layer_names[:max_layers]

    for layer_index, layer_name in enumerate(shared_layer_names):
        clean_heatmap = summarize_feature_map(clean_features[layer_name], target_size=target_size)
        adversarial_heatmap = summarize_feature_map(adversarial_features[layer_name], target_size=target_size)
        diff_heatmap = np.abs(adversarial_heatmap - clean_heatmap)

        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        panels = (
            ("Clean", colorize_heatmap(clean_heatmap, cmap_name=cmap_name)),
            ("Adversarial", colorize_heatmap(adversarial_heatmap, cmap_name=cmap_name)),
            ("Abs Diff", colorize_heatmap(diff_heatmap, cmap_name=cmap_name)),
        )
        for axis, (title, image) in zip(axes, panels, strict=True):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(f"{sample_key} | {layer_name} | clean_shape={tuple(clean_features[layer_name].shape)}", fontsize=11)
        figure.tight_layout(rect=(0.0, 0.03, 1.0, 0.92))

        figure_path = sample_dir / f"{layer_index:02d}_{_sanitize_name(layer_name)}.png"
        figure.savefig(figure_path, dpi=160, bbox_inches="tight")
        plt.close(figure)

        layer_records.append(
            {
                "layer_name": layer_name,
                "figure_path": str(figure_path.resolve()),
                "clean_shape": list(clean_features[layer_name].shape),
                "adversarial_shape": list(adversarial_features[layer_name].shape),
            }
        )

    metadata = {
        "sample_key": sample_key,
        "sample_dir": str(sample_dir.resolve()),
        "input_path": str((sample_dir / "input.png").resolve()),
        "adversarial_path": str((sample_dir / "adversarial.png").resolve()),
        "perturbation_path": str((sample_dir / "perturbation.png").resolve()),
        "layers": layer_records,
    }
    (sample_dir / "feature_maps.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata
