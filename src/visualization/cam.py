from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.models import TorchSegmentationModelAdapter
from src.models.architectures.segmenter import SegMenter, pad_to_patch_size, remove_padding
from src.robustness.visualization import normalize_heatmap, overlay_heatmap_on_image, resolve_heatmap_display_bounds


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
    clean_pred_pixels: int
    adversarial_pred_pixels: int
    clean_top20_area_ratio: float
    adversarial_top20_area_ratio: float
    clean_inside_gt_ratio: float
    adversarial_inside_gt_ratio: float
    clean_inside_clean_prediction_ratio: float
    adversarial_inside_clean_prediction_ratio: float
    centroid_shift: float | None
    clean_used_fallback: bool
    adversarial_used_fallback: bool


def discover_cam_supported_feature_keys(
    model: TorchSegmentationModelAdapter,
    feature_keys: list[str],
) -> list[str]:
    # Prefer feature-name patterns over strict type checks. This is robust to
    # Streamlit hot-reload / cache reuse, where stale model instances can fail
    # isinstance(...) checks after code reload even though their feature maps
    # remain valid for CAM.
    backbone_stage_keys = [key for key in feature_keys if key.startswith("backbone:stage")]
    if backbone_stage_keys:
        return backbone_stage_keys

    encoder_block_keys = [key for key in feature_keys if key.startswith("encoder:block")]
    if encoder_block_keys:
        return encoder_block_keys

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


def select_representative_cam_feature_keys(
    model: TorchSegmentationModelAdapter,
    feature_keys: list[str],
    max_keys: int = 3,
) -> list[str]:
    supported_keys = discover_cam_supported_feature_keys(model, feature_keys)
    if max_keys <= 0 or not supported_keys:
        return []
    if len(supported_keys) <= max_keys:
        return supported_keys

    target_count = min(max_keys, len(supported_keys))
    if target_count == 1:
        return [supported_keys[-1]]

    selected_keys: list[str] = []
    for index in range(target_count):
        position = round(index * (len(supported_keys) - 1) / (target_count - 1))
        feature_key = supported_keys[position]
        if feature_key not in selected_keys:
            selected_keys.append(feature_key)

    if len(selected_keys) < target_count:
        for feature_key in supported_keys:
            if feature_key in selected_keys:
                continue
            selected_keys.append(feature_key)
            if len(selected_keys) == target_count:
                break

    return selected_keys


def compute_feature_grad_cam(
    model: TorchSegmentationModelAdapter,
    image: torch.Tensor,
    feature_key: str,
    class_id: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if feature_key.startswith("encoder:block") and _supports_segmenter_cam(model.model):
        return _compute_segmenter_grad_cam(model, image, feature_key, class_id)

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
        "used_fallback": target_pixels == 0,
    }


def _compute_segmenter_grad_cam(
    model: TorchSegmentationModelAdapter,
    image: torch.Tensor,
    feature_key: str,
    class_id: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    model.model.zero_grad(set_to_none=True)
    inputs = image.unsqueeze(0).to(model.device)
    segmenter = model.model

    if not _supports_segmenter_cam(segmenter):
        raise TypeError("Segmenter CAM path expects a Segmenter-like model with encoder/decoder/patch_size/backbone.")

    try:
        block_index = int(feature_key.replace("encoder:block", ""))
    except ValueError as exc:
        raise KeyError(f"Unsupported Segmenter CAM feature key: {feature_key!r}") from exc

    original_size = (inputs.size(2), inputs.size(3))
    padded = pad_to_patch_size(inputs, segmenter.patch_size)
    padded_size = (padded.size(2), padded.size(3))
    grid_h = padded_size[0] // segmenter.patch_size
    grid_w = padded_size[1] // segmenter.patch_size
    num_extra_tokens = 0 if "SAM" in segmenter.backbone else 1 + int(segmenter.encoder.distilled)

    with torch.enable_grad():
        tokens, hidden_states = segmenter.encoder.forward_tokens(padded, collect_hidden_states=True)
        if block_index >= len(hidden_states):
            raise KeyError(
                f"Feature key '{feature_key}' is not available for CAM. "
                f"Segmenter returned {len(hidden_states)} encoder blocks."
            )

        hidden_state = hidden_states[block_index]
        spatial_tokens = hidden_state[:, num_extra_tokens:]
        feature_map = spatial_tokens.transpose(1, 2).reshape(spatial_tokens.size(0), spatial_tokens.size(2), grid_h, grid_w)

        logits = segmenter.decoder(tokens[:, num_extra_tokens:], padded_size)
        logits = F.interpolate(logits, size=padded_size, mode="bilinear")
        logits = remove_padding(logits, original_size)

        class_logits = logits[:, int(class_id)]
        prediction = logits.argmax(dim=1)
        target_mask = prediction == int(class_id)
        target_pixels = int(target_mask.sum().detach().cpu().item())
        if target_pixels > 0:
            target_score = class_logits.masked_select(target_mask).mean()
        else:
            target_score = class_logits.mean()

        # Segmenter feature maps are reshaped views of token sequences. The
        # decoder graph consumes the original hidden-state tensor, not the 4D
        # view, so gradients must be taken w.r.t. the hidden state and then
        # reshaped back to the spatial patch map for CAM.
        hidden_gradients = torch.autograd.grad(
            target_score,
            hidden_state,
            retain_graph=False,
            create_graph=False,
        )[0]
        gradients = hidden_gradients[:, num_extra_tokens:].transpose(1, 2).reshape_as(feature_map)

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
        "used_fallback": target_pixels == 0,
    }


def _supports_segmenter_cam(model: object) -> bool:
    return all(
        hasattr(model, attribute)
        for attribute in ("encoder", "decoder", "patch_size", "backbone")
    )


def _build_top_activation_mask(heatmap: np.ndarray, top_percent: int = 20) -> np.ndarray:
    if heatmap.ndim != 2:
        raise ValueError(f"Expected heatmap with shape [H, W], got {heatmap.shape}.")
    if heatmap.size == 0:
        return np.zeros_like(heatmap, dtype=bool)
    if float(heatmap.max()) <= 0.0:
        return np.zeros_like(heatmap, dtype=bool)

    clipped_top_percent = int(np.clip(int(top_percent), 0, 100))
    threshold_percentile = max(0, 100 - clipped_top_percent)
    threshold = float(np.percentile(heatmap, threshold_percentile))
    return heatmap >= threshold


def _mask_inside_ratio(region_mask: np.ndarray, reference_mask: np.ndarray) -> float:
    region = np.asarray(region_mask, dtype=bool)
    reference = np.asarray(reference_mask, dtype=bool)
    if region.shape != reference.shape:
        raise ValueError(f"Mask shape mismatch: {region.shape} vs {reference.shape}.")
    region_area = int(region.sum())
    if region_area == 0:
        return 0.0
    return float(np.logical_and(region, reference).sum() / region_area)


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    points = np.argwhere(np.asarray(mask, dtype=bool))
    if points.size == 0:
        return None
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def _centroid_shift(left_mask: np.ndarray, right_mask: np.ndarray) -> float | None:
    left_center = _mask_centroid(left_mask)
    right_center = _mask_centroid(right_mask)
    if left_center is None or right_center is None:
        return None
    left_array = np.asarray(left_center, dtype=np.float32)
    right_array = np.asarray(right_center, dtype=np.float32)
    return float(np.linalg.norm(left_array - right_array))


def build_cam_visualization(
    model: TorchSegmentationModelAdapter,
    clean_tensor: torch.Tensor,
    adversarial_tensor: torch.Tensor,
    clean_image: np.ndarray,
    adversarial_image: np.ndarray,
    feature_key: str,
    class_id: int,
    ground_truth: np.ndarray | None = None,
    clean_prediction: np.ndarray | None = None,
    heatmap_scale_mode: str = "independent",
    heatmap_percentile_clip_upper: float = 100.0,
) -> CamVisualizationResult:
    clean_heatmap, clean_metadata = compute_feature_grad_cam(model, clean_tensor, feature_key, class_id)
    adversarial_heatmap, adversarial_metadata = compute_feature_grad_cam(model, adversarial_tensor, feature_key, class_id)
    diff_heatmap = normalize_heatmap(np.abs(adversarial_heatmap - clean_heatmap))
    display_bounds = resolve_heatmap_display_bounds(
        [clean_heatmap, adversarial_heatmap, diff_heatmap],
        scale_mode=heatmap_scale_mode,
        percentile_clip_upper=heatmap_percentile_clip_upper,
    )

    clean_top20_mask = _build_top_activation_mask(clean_heatmap, top_percent=20)
    adversarial_top20_mask = _build_top_activation_mask(adversarial_heatmap, top_percent=20)

    target_ground_truth_mask = (
        np.asarray(ground_truth, dtype=np.int64) == int(class_id)
        if ground_truth is not None
        else np.zeros_like(clean_heatmap, dtype=bool)
    )
    target_clean_prediction_mask = (
        np.asarray(clean_prediction, dtype=np.int64) == int(class_id)
        if clean_prediction is not None
        else np.zeros_like(clean_heatmap, dtype=bool)
    )

    return CamVisualizationResult(
        feature_key=feature_key,
        class_id=int(class_id),
        clean_overlay=overlay_heatmap_on_image(
            clean_image,
            clean_heatmap,
            alpha=0.45,
            cmap_name="jet",
            vmin=display_bounds[0][0],
            vmax=display_bounds[0][1],
        ),
        adversarial_overlay=overlay_heatmap_on_image(
            adversarial_image,
            adversarial_heatmap,
            alpha=0.45,
            cmap_name="jet",
            vmin=display_bounds[1][0],
            vmax=display_bounds[1][1],
        ),
        diff_image=overlay_heatmap_on_image(
            adversarial_image,
            diff_heatmap,
            alpha=0.50,
            cmap_name="inferno",
            vmin=display_bounds[2][0],
            vmax=display_bounds[2][1],
        ),
        clean_mean=float(clean_heatmap.mean()),
        adversarial_mean=float(adversarial_heatmap.mean()),
        diff_mean=float(diff_heatmap.mean()),
        clean_pred_pixels=int(clean_metadata["target_pixels"]),
        adversarial_pred_pixels=int(adversarial_metadata["target_pixels"]),
        clean_top20_area_ratio=float(clean_top20_mask.mean()) if clean_top20_mask.size else 0.0,
        adversarial_top20_area_ratio=float(adversarial_top20_mask.mean()) if adversarial_top20_mask.size else 0.0,
        clean_inside_gt_ratio=_mask_inside_ratio(clean_top20_mask, target_ground_truth_mask),
        adversarial_inside_gt_ratio=_mask_inside_ratio(adversarial_top20_mask, target_ground_truth_mask),
        clean_inside_clean_prediction_ratio=_mask_inside_ratio(clean_top20_mask, target_clean_prediction_mask),
        adversarial_inside_clean_prediction_ratio=_mask_inside_ratio(adversarial_top20_mask, target_clean_prediction_mask),
        centroid_shift=_centroid_shift(clean_top20_mask, adversarial_top20_mask),
        clean_used_fallback=bool(clean_metadata["used_fallback"]),
        adversarial_used_fallback=bool(adversarial_metadata["used_fallback"]),
    )
