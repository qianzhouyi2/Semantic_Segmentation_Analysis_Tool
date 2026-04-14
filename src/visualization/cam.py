from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.models import TorchSegmentationModelAdapter
from src.models.architectures.segmenter import SegMenter
from src.models.architectures.upernet import TorchvisionResNetBackbone, UperNetForSemanticSegmentation
from src.models.backbones.convnext import ConvNeXt
from src.robustness.visualization import normalize_heatmap, overlay_heatmap_on_image


@dataclass(slots=True)
class CamVisualizationResult:
    feature_key: str
    class_id: int
    clean_overlay: np.ndarray
    adversarial_overlay: np.ndarray
    diff_image: np.ndarray
    clean_mean: float
    adversarial_mean: float
    diff_mean: float
    clean_target_pixels: int
    adversarial_target_pixels: int


def discover_cam_supported_feature_keys(
    model: TorchSegmentationModelAdapter,
    feature_keys: list[str],
) -> list[str]:
    if isinstance(model.model, UperNetForSemanticSegmentation):
        if isinstance(model.model.backbone, ConvNeXt):
            return [key for key in feature_keys if key.startswith("backbone:stage")]
        if isinstance(model.model.backbone, TorchvisionResNetBackbone):
            return [key for key in feature_keys if key.startswith("backbone:stage")]
    if isinstance(model.model, SegMenter):
        return [key for key in feature_keys if key.startswith("encoder:block")]
    return [key for key in feature_keys if key == "logits"]


def select_default_cam_feature_key(
    model: TorchSegmentationModelAdapter,
    feature_keys: list[str],
) -> str | None:
    supported_keys = discover_cam_supported_feature_keys(model, feature_keys)
    if not supported_keys:
        return None
    # Conventional CAM usage attaches to the deepest available semantic layer.
    return supported_keys[-1]


def compute_feature_grad_cam(
    model: TorchSegmentationModelAdapter,
    image: torch.Tensor,
    feature_key: str,
    class_id: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    model.model.zero_grad(set_to_none=True)
    inputs = image.unsqueeze(0).to(model.device)

    with torch.enable_grad():
        logits, features = model.forward_with_features(inputs)
        if feature_key not in features:
            raise KeyError(f"Feature key '{feature_key}' is not available for CAM.")

        feature_map = features[feature_key]
        class_logits = logits[:, int(class_id)]
        prediction = logits.argmax(dim=1)
        target_mask = prediction == int(class_id)
        target_pixels = int(target_mask.sum().detach().cpu().item())
        if target_pixels > 0:
            target_score = class_logits.masked_select(target_mask).mean()
        else:
            target_score = class_logits.mean()

        gradients = torch.autograd.grad(target_score, feature_map, retain_graph=False, create_graph=False)[0]

    if feature_map.ndim != 4 or gradients.ndim != 4:
        raise ValueError(
            f"CAM currently expects 4D feature maps. Got feature shape={tuple(feature_map.shape)} "
            f"and gradient shape={tuple(gradients.shape)}."
        )

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
    cam_heatmap = normalize_heatmap(cam[0, 0].detach().cpu().numpy())
    model.model.zero_grad(set_to_none=True)
    return cam_heatmap, {
        "feature_key": feature_key,
        "class_id": int(class_id),
        "target_pixels": target_pixels,
        "score": float(target_score.detach().cpu().item()),
    }


def build_cam_visualization(
    model: TorchSegmentationModelAdapter,
    clean_tensor: torch.Tensor,
    adversarial_tensor: torch.Tensor,
    clean_image: np.ndarray,
    adversarial_image: np.ndarray,
    feature_key: str,
    class_id: int,
) -> CamVisualizationResult:
    clean_heatmap, clean_metadata = compute_feature_grad_cam(model, clean_tensor, feature_key, class_id)
    adversarial_heatmap, adversarial_metadata = compute_feature_grad_cam(model, adversarial_tensor, feature_key, class_id)
    diff_heatmap = normalize_heatmap(np.abs(adversarial_heatmap - clean_heatmap))

    return CamVisualizationResult(
        feature_key=feature_key,
        class_id=int(class_id),
        clean_overlay=overlay_heatmap_on_image(clean_image, clean_heatmap, alpha=0.45, cmap_name="jet"),
        adversarial_overlay=overlay_heatmap_on_image(adversarial_image, adversarial_heatmap, alpha=0.45, cmap_name="jet"),
        diff_image=overlay_heatmap_on_image(adversarial_image, diff_heatmap, alpha=0.50, cmap_name="inferno"),
        clean_mean=float(clean_heatmap.mean()),
        adversarial_mean=float(adversarial_heatmap.mean()),
        diff_mean=float(diff_heatmap.mean()),
        clean_target_pixels=int(clean_metadata["target_pixels"]),
        adversarial_target_pixels=int(adversarial_metadata["target_pixels"]),
    )
