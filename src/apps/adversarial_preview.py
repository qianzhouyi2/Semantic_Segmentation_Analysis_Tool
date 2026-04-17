from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.attacks import AttackConfig, AttackOutput, AttackRunner
from src.common.config import load_yaml, resolve_project_path
from src.models import TorchSegmentationModelAdapter, load_sparse_defense_config
from src.robustness.visualization import (
    colorize_heatmap,
    normalize_perturbation,
    resolve_heatmap_display_bounds,
    summarize_image_delta,
    summarize_feature_map,
    tensor_to_rgb_image,
)


CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt")
KNOWN_MODEL_CHECKPOINTS = (
    ("UperNet ConvNext T VOC adv", "upernet_convnext", Path("models/UperNet_ConvNext_T_VOC_adv.pth")),
    ("UperNet ConvNext T VOC clean", "upernet_convnext", Path("models/UperNet_ConvNext_T_VOC_clean.pth")),
    ("UperNet ResNet50 VOC adv", "upernet_resnet50", Path("models/UperNet_ResNet50_VOC_adv.pth")),
    ("UperNet ResNet50 VOC clean", "upernet_resnet50", Path("models/UperNet_ResNet50_VOC_clean.pth")),
    ("Segmenter ViT-S VOC adv", "segmenter_vit_s", Path("models/Segmenter_ViT_S_VOC_adv.pth")),
    ("Segmenter ViT-S VOC clean", "segmenter_vit_s", Path("models/Segmenter_ViT_S_VOC_clean.pth")),
)


@dataclass(slots=True, frozen=True)
class AttackConfigOption:
    attack_name: str
    label: str
    path: Path


@dataclass(slots=True, frozen=True)
class CheckpointOption:
    family: str
    label: str
    path: Path


@dataclass(slots=True, frozen=True)
class DefenseConfigOption:
    variant: str
    label: str
    path: Path
    family: str | None
    threshold: float


@dataclass(slots=True)
class FeaturePreviewResult:
    sample_id: str
    attack_name: str
    epsilon: float
    step_size: float
    steps: int
    clean_tensor: torch.Tensor
    adversarial_tensor: torch.Tensor
    clean_image: np.ndarray
    adversarial_image: np.ndarray
    perturbation_image: np.ndarray
    sample_delta_map: np.ndarray
    sample_delta_heatmap: np.ndarray
    sample_delta_mean: float
    sample_delta_max: float
    ground_truth: np.ndarray
    clean_prediction: np.ndarray
    adversarial_prediction: np.ndarray
    clean_features: dict[str, torch.Tensor]
    adversarial_features: dict[str, torch.Tensor]
    layer_names: list[str]
    attack_metadata: dict[str, Any]


def infer_model_family_from_checkpoint(path: str | Path) -> str | None:
    stem = Path(path).stem.lower()
    if "convnext" in stem:
        return "upernet_convnext"
    if "resnet50" in stem or "resnet_50" in stem or "resnet-50" in stem:
        return "upernet_resnet50"
    if "segmenter" in stem or "vit_s" in stem or "vit-small" in stem or "vit_small" in stem:
        return "segmenter_vit_s"
    return None


def discover_attack_config_options(config_dir: str | Path = "configs/attacks") -> list[AttackConfigOption]:
    directory = resolve_project_path(config_dir)
    options: list[AttackConfigOption] = []
    for path in sorted(directory.glob("*.y*ml")):
        try:
            config = AttackConfig.from_dict(load_yaml(path))
        except (KeyError, TypeError, ValueError):
            continue
        options.append(
            AttackConfigOption(
                attack_name=config.name,
                label=f"{path.stem} ({config.name})",
                path=path.resolve(),
            )
        )
    options.sort(key=lambda item: (item.attack_name, item.path.stem))
    return options


def discover_checkpoint_options(models_dir: str | Path = "models", include_known: bool = True) -> list[CheckpointOption]:
    directory = resolve_project_path(models_dir)
    options: list[CheckpointOption] = []
    seen_paths: set[Path] = set()

    if include_known:
        for label, family, raw_path in KNOWN_MODEL_CHECKPOINTS:
            path = resolve_project_path(raw_path)
            if path.exists():
                resolved = path.resolve()
                options.append(CheckpointOption(family=family, label=label, path=resolved))
                seen_paths.add(resolved)

    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in CHECKPOINT_SUFFIXES or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        family = infer_model_family_from_checkpoint(resolved)
        if family is None:
            continue
        options.append(CheckpointOption(family=family, label=path.stem, path=resolved))
        seen_paths.add(resolved)

    options.sort(key=lambda item: (item.family, item.label.lower()))
    return options


def discover_defense_config_options(config_dir: str | Path = "configs/defenses") -> list[DefenseConfigOption]:
    directory = resolve_project_path(config_dir)
    options: list[DefenseConfigOption] = []
    if not directory.exists():
        return options

    for path in sorted(directory.glob("*.y*ml")):
        try:
            config = load_sparse_defense_config(path)
        except (TypeError, ValueError):
            continue
        family_suffix = "" if config.family is None else f", family={config.family}"
        options.append(
            DefenseConfigOption(
                variant=config.variant,
                label=f"{path.stem} ({config.variant}, thr={config.threshold:.2f}{family_suffix})",
                path=path.resolve(),
                family=config.family,
                threshold=float(config.threshold),
            )
        )

    options.sort(key=lambda item: (item.variant, item.path.stem))
    return options


def ordered_feature_layer_names(features: dict[str, torch.Tensor]) -> list[str]:
    def _sort_backbone_block(name: str) -> tuple[int, int]:
        _, stage_part, block_part = name.split(":")
        return int(stage_part.replace("stage", "")), int(block_part.replace("block", ""))

    def _sort_encoder_block(name: str) -> int:
        _, block_part = name.split(":")
        return int(block_part.replace("block", ""))

    backbone_block_names = [name for name in features if name.count(":") == 2 and name.startswith("backbone:stage")]
    if backbone_block_names:
        return sorted(backbone_block_names, key=_sort_backbone_block)

    encoder_block_names = [name for name in features if name.startswith("encoder:block")]
    if encoder_block_names:
        return sorted(encoder_block_names, key=_sort_encoder_block)

    backbone_stage_names = [name for name in features if name.count(":") == 1 and name.startswith("backbone:stage")]
    if backbone_stage_names:
        return sorted(backbone_stage_names)

    excluded = {"encoder", "encoder:first", "encoder:last", "backbone:first", "backbone:last"}
    ordered = [name for name in features if name not in excluded]
    return ordered or list(features)


def _to_cpu_feature_dict(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: feature.detach().cpu() for name, feature in features.items()}


def _perturbation_to_image(perturbation: torch.Tensor) -> np.ndarray:
    data = perturbation.detach().float().cpu()
    if data.ndim == 4:
        data = data[0]
    return normalize_perturbation(data.permute(1, 2, 0).numpy())


def heatmap_display_note(scale_mode: str, percentile_clip_upper: float) -> str:
    clipped_percentile = float(np.clip(percentile_clip_upper, 0.0, 100.0))
    if scale_mode == "fixed":
        note = "fixed-scale normalized heatmap [0,1]"
    elif scale_mode == "shared":
        note = "shared-scale normalized heatmap"
    else:
        note = "independent normalized heatmap"
    if clipped_percentile < 100.0:
        return f"{note}, clip<=P{clipped_percentile:.0f}"
    return note


def build_sample_delta_visualization(
    result: FeaturePreviewResult,
    *,
    scale_mode: str = "independent",
    percentile_clip_upper: float = 100.0,
) -> dict[str, Any]:
    delta_heatmap = result.sample_delta_map
    display_bounds = resolve_heatmap_display_bounds(
        [delta_heatmap],
        scale_mode=scale_mode,
        percentile_clip_upper=percentile_clip_upper,
    )[0]
    return {
        "delta_image": colorize_heatmap(
            delta_heatmap,
            cmap_name="inferno",
            vmin=display_bounds[0],
            vmax=display_bounds[1],
        ),
        "mean_abs_delta": float(delta_heatmap.mean()),
        "max_abs_delta": float(delta_heatmap.max()),
        "display_note": heatmap_display_note(scale_mode, percentile_clip_upper),
        "display_range": display_bounds,
    }


def generate_feature_preview(
    model: TorchSegmentationModelAdapter,
    attack_config: AttackConfig,
    image: torch.Tensor,
    target: torch.Tensor,
    sample_id: str,
) -> FeaturePreviewResult:
    attack_runner = AttackRunner(model)
    images = image.unsqueeze(0).to(model.device)
    targets = target.unsqueeze(0).to(model.device)

    with torch.no_grad():
        clean_logits, clean_features = model.forward_with_features(images)
        clean_prediction = clean_logits.argmax(dim=1)

    if attack_config.epsilon <= 0:
        attack_output = AttackOutput(
            adversarial_images=images.detach().clone(),
            perturbation=torch.zeros_like(images),
            metadata={
                "attack": attack_config.name,
                "loss": 0.0,
                "steps": 0,
                "step_size": 0.0,
                "skipped": True,
                "reason": "zero_radius",
            },
        )
    else:
        attack_output = attack_runner.run(config=attack_config, images=images, targets=targets)

    with torch.no_grad():
        adversarial_logits, adversarial_features = model.forward_with_features(attack_output.adversarial_images)
        adversarial_prediction = adversarial_logits.argmax(dim=1)

    clean_features_cpu = _to_cpu_feature_dict(clean_features)
    adversarial_features_cpu = _to_cpu_feature_dict(adversarial_features)
    layer_names = ordered_feature_layer_names(clean_features_cpu)
    layer_names = [name for name in layer_names if name in adversarial_features_cpu]
    sample_delta = summarize_image_delta(images, attack_output.adversarial_images)

    return FeaturePreviewResult(
        sample_id=sample_id,
        attack_name=attack_config.name,
        epsilon=float(attack_config.epsilon),
        step_size=float(attack_config.resolved_step_size()),
        steps=int(attack_config.steps),
        clean_tensor=images[0].detach().cpu(),
        adversarial_tensor=attack_output.adversarial_images[0].detach().cpu(),
        clean_image=tensor_to_rgb_image(images),
        adversarial_image=tensor_to_rgb_image(attack_output.adversarial_images),
        perturbation_image=_perturbation_to_image(attack_output.perturbation),
        sample_delta_map=sample_delta,
        sample_delta_heatmap=colorize_heatmap(sample_delta, cmap_name="inferno"),
        sample_delta_mean=float(sample_delta.mean()),
        sample_delta_max=float(sample_delta.max()),
        ground_truth=targets[0].detach().cpu().numpy(),
        clean_prediction=clean_prediction[0].detach().cpu().numpy(),
        adversarial_prediction=adversarial_prediction[0].detach().cpu().numpy(),
        clean_features=clean_features_cpu,
        adversarial_features=adversarial_features_cpu,
        layer_names=layer_names,
        attack_metadata=dict(attack_output.metadata),
    )


def compute_layer_heatmaps(
    result: FeaturePreviewResult,
    layer_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_heatmap = summarize_feature_map(
        result.clean_features[layer_name],
        target_size=result.clean_image.shape[:2],
    )
    adversarial_heatmap = summarize_feature_map(
        result.adversarial_features[layer_name],
        target_size=result.clean_image.shape[:2],
    )
    diff_heatmap = np.abs(adversarial_heatmap - clean_heatmap)
    return clean_heatmap, adversarial_heatmap, diff_heatmap


def build_layer_visualization(
    result: FeaturePreviewResult,
    layer_name: str,
    *,
    scale_mode: str = "independent",
    percentile_clip_upper: float = 100.0,
) -> dict[str, Any]:
    clean_heatmap, adversarial_heatmap, diff_heatmap = compute_layer_heatmaps(result, layer_name)
    display_bounds = resolve_heatmap_display_bounds(
        [clean_heatmap, adversarial_heatmap, diff_heatmap],
        scale_mode=scale_mode,
        percentile_clip_upper=percentile_clip_upper,
    )

    return {
        "layer_name": layer_name,
        "clean_heatmap": clean_heatmap,
        "adversarial_heatmap": adversarial_heatmap,
        "diff_heatmap": diff_heatmap,
        "clean_image": colorize_heatmap(
            clean_heatmap,
            vmin=display_bounds[0][0],
            vmax=display_bounds[0][1],
        ),
        "adversarial_image": colorize_heatmap(
            adversarial_heatmap,
            vmin=display_bounds[1][0],
            vmax=display_bounds[1][1],
        ),
        "diff_image": colorize_heatmap(
            diff_heatmap,
            vmin=display_bounds[2][0],
            vmax=display_bounds[2][1],
        ),
        "clean_shape": tuple(result.clean_features[layer_name].shape),
        "adversarial_shape": tuple(result.adversarial_features[layer_name].shape),
        "mean_abs_diff": float(diff_heatmap.mean()),
        "max_abs_diff": float(diff_heatmap.max()),
        "display_note": heatmap_display_note(scale_mode, percentile_clip_upper),
        "display_ranges": display_bounds,
    }


def select_representative_layer_names(layer_names: list[str], max_layers: int = 3) -> list[str]:
    if max_layers <= 0 or not layer_names:
        return []
    if len(layer_names) <= max_layers:
        return list(layer_names)

    selected: list[str] = []
    target_count = min(max_layers, len(layer_names))
    for index in range(target_count):
        position = round(index * (len(layer_names) - 1) / (target_count - 1))
        layer_name = layer_names[position]
        if layer_name not in selected:
            selected.append(layer_name)

    if len(selected) < target_count:
        for layer_name in layer_names:
            if layer_name in selected:
                continue
            selected.append(layer_name)
            if len(selected) == target_count:
                break
    return selected


def run_feature_preview_sweep(
    model: TorchSegmentationModelAdapter,
    attack_config: AttackConfig,
    image: torch.Tensor,
    target: torch.Tensor,
    sample_id: str,
    radii_255: list[int],
) -> list[FeaturePreviewResult]:
    unique_radii = sorted({int(radius) for radius in radii_255 if 0 <= int(radius) <= 255})
    return [
        generate_feature_preview(
            model=model,
            attack_config=attack_config.with_radius_255(radius_255),
            image=image,
            target=target,
            sample_id=sample_id,
        )
        for radius_255 in unique_radii
    ]
