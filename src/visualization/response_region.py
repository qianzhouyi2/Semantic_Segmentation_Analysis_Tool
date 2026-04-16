from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.models import TorchSegmentationModelAdapter
from src.robustness.visualization import (
    normalize_heatmap,
    overlay_binary_mask_on_image,
    overlay_heatmap_on_image,
)


@dataclass(slots=True)
class ResponseRegionVisualizationResult:
    class_id: int
    threshold_percentile: int
    clean_heatmap: np.ndarray
    adversarial_heatmap: np.ndarray
    diff_heatmap: np.ndarray
    clean_region_mask: np.ndarray
    adversarial_region_mask: np.ndarray
    overlap_region_mask: np.ndarray
    clean_overlay: np.ndarray
    adversarial_overlay: np.ndarray
    diff_overlay: np.ndarray
    clean_region_overlay: np.ndarray
    adversarial_region_overlay: np.ndarray
    overlap_region_overlay: np.ndarray
    clean_mean: float
    adversarial_mean: float
    diff_mean: float
    clean_peak: float
    adversarial_peak: float
    clean_active_ratio: float
    adversarial_active_ratio: float
    overlap_iou: float
    clean_target_pixels: int
    adversarial_target_pixels: int
    clean_score: float
    adversarial_score: float


def _target_score_from_class_logits(logits: torch.Tensor, class_id: int) -> tuple[torch.Tensor, int]:
    class_logits = logits[:, int(class_id)]
    prediction = logits.argmax(dim=1)
    target_mask = prediction == int(class_id)
    target_pixels = int(target_mask.sum().detach().cpu().item())
    if target_pixels > 0:
        return class_logits.masked_select(target_mask).mean(), target_pixels
    return class_logits.mean(), target_pixels


def compute_input_response_heatmap(
    model: TorchSegmentationModelAdapter,
    image: torch.Tensor,
    class_id: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    model.model.zero_grad(set_to_none=True)
    inputs = image.unsqueeze(0).detach().to(model.device).clone().requires_grad_(True)

    with torch.enable_grad():
        logits = model.logits(inputs)
        target_score, target_pixels = _target_score_from_class_logits(logits, class_id)
        gradients = torch.autograd.grad(target_score, inputs, retain_graph=False, create_graph=False)[0]

    response_map = gradients.abs().mean(dim=1)[0].detach().cpu().numpy()
    heatmap = normalize_heatmap(response_map)
    model.model.zero_grad(set_to_none=True)
    return heatmap, {
        "class_id": int(class_id),
        "target_pixels": target_pixels,
        "score": float(target_score.detach().cpu().item()),
        "mean_response": float(response_map.mean()),
        "peak_response": float(response_map.max()) if response_map.size else 0.0,
    }


def _build_response_region_mask(heatmap: np.ndarray, threshold_percentile: int) -> np.ndarray:
    if heatmap.ndim != 2:
        raise ValueError(f"Expected heatmap with shape [H, W], got {heatmap.shape}.")
    if heatmap.size == 0:
        return np.zeros_like(heatmap, dtype=bool)
    if float(heatmap.max()) <= 0.0:
        return np.zeros_like(heatmap, dtype=bool)

    clipped_percentile = int(np.clip(int(threshold_percentile), 0, 100))
    threshold = float(np.percentile(heatmap, clipped_percentile))
    return heatmap >= threshold


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_mask = np.asarray(left, dtype=bool)
    right_mask = np.asarray(right, dtype=bool)
    intersection = int(np.logical_and(left_mask, right_mask).sum())
    union = int(np.logical_or(left_mask, right_mask).sum())
    if union == 0:
        return 0.0
    return float(intersection / union)


def build_response_region_visualization(
    model: TorchSegmentationModelAdapter,
    clean_tensor: torch.Tensor,
    adversarial_tensor: torch.Tensor,
    clean_image: np.ndarray,
    adversarial_image: np.ndarray,
    class_id: int,
    threshold_percentile: int = 85,
) -> ResponseRegionVisualizationResult:
    clean_heatmap, clean_metadata = compute_input_response_heatmap(model, clean_tensor, class_id)
    adversarial_heatmap, adversarial_metadata = compute_input_response_heatmap(model, adversarial_tensor, class_id)
    diff_heatmap = normalize_heatmap(np.abs(adversarial_heatmap - clean_heatmap))

    clean_region_mask = _build_response_region_mask(clean_heatmap, threshold_percentile)
    adversarial_region_mask = _build_response_region_mask(adversarial_heatmap, threshold_percentile)
    overlap_region_mask = np.logical_and(clean_region_mask, adversarial_region_mask)

    return ResponseRegionVisualizationResult(
        class_id=int(class_id),
        threshold_percentile=int(threshold_percentile),
        clean_heatmap=clean_heatmap,
        adversarial_heatmap=adversarial_heatmap,
        diff_heatmap=diff_heatmap,
        clean_region_mask=clean_region_mask,
        adversarial_region_mask=adversarial_region_mask,
        overlap_region_mask=overlap_region_mask,
        clean_overlay=overlay_heatmap_on_image(clean_image, clean_heatmap, alpha=0.45, cmap_name="jet"),
        adversarial_overlay=overlay_heatmap_on_image(
            adversarial_image,
            adversarial_heatmap,
            alpha=0.45,
            cmap_name="jet",
        ),
        diff_overlay=overlay_heatmap_on_image(adversarial_image, diff_heatmap, alpha=0.50, cmap_name="inferno"),
        clean_region_overlay=overlay_binary_mask_on_image(clean_image, clean_region_mask, color=(255, 128, 0), alpha=0.55),
        adversarial_region_overlay=overlay_binary_mask_on_image(
            adversarial_image,
            adversarial_region_mask,
            color=(0, 196, 255),
            alpha=0.55,
        ),
        overlap_region_overlay=overlay_binary_mask_on_image(clean_image, overlap_region_mask, color=(32, 224, 64), alpha=0.60),
        clean_mean=float(clean_heatmap.mean()),
        adversarial_mean=float(adversarial_heatmap.mean()),
        diff_mean=float(diff_heatmap.mean()),
        clean_peak=float(clean_heatmap.max()) if clean_heatmap.size else 0.0,
        adversarial_peak=float(adversarial_heatmap.max()) if adversarial_heatmap.size else 0.0,
        clean_active_ratio=float(clean_region_mask.mean()) if clean_region_mask.size else 0.0,
        adversarial_active_ratio=float(adversarial_region_mask.mean()) if adversarial_region_mask.size else 0.0,
        overlap_iou=_mask_iou(clean_region_mask, adversarial_region_mask),
        clean_target_pixels=int(clean_metadata["target_pixels"]),
        adversarial_target_pixels=int(adversarial_metadata["target_pixels"]),
        clean_score=float(clean_metadata["score"]),
        adversarial_score=float(adversarial_metadata["score"]),
    )
