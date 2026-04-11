from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.architectures import UperNetForSemanticSegmentation, build_segmenter_vit_small_patch16_224


MODEL_FAMILY_CHOICES = (
    "upernet_convnext",
    "upernet_resnet50",
    "segmenter_vit_s",
)


def build_model(family: str, num_classes: int = 21) -> nn.Module:
    if family == "upernet_convnext":
        return UperNetForSemanticSegmentation("ConvNeXt-T_CVST", n_cls=num_classes, pretrained=None)
    if family == "upernet_resnet50":
        return UperNetForSemanticSegmentation("ResNet-50", n_cls=num_classes, pretrained=None)
    if family == "segmenter_vit_s":
        return build_segmenter_vit_small_patch16_224(num_classes=num_classes)
    raise ValueError(f"Unknown model family: {family}. Expected one of {MODEL_FAMILY_CHOICES}.")


def _extract_state_dict(payload: Any) -> OrderedDict[str, torch.Tensor]:
    if isinstance(payload, OrderedDict):
        return payload
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, (dict, OrderedDict)):
                return OrderedDict(value)
        return OrderedDict(payload)
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def normalize_state_dict_keys(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        normalized_key = key
        if normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]
        if normalized_key.startswith("model."):
            normalized_key = normalized_key[len("model.") :]
        if normalized_key.startswith("base_model."):
            normalized_key = normalized_key[len("base_model.") :]
        if normalized_key.startswith("base_") and not normalized_key.startswith("base_normalize."):
            normalized_key = normalized_key[len("base_") :]
        if normalized_key in {"base_normalize.mean", "base_normalize.std"}:
            continue
        normalized[normalized_key] = value
    return normalized


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = normalize_state_dict_keys(_extract_state_dict(checkpoint))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    return list(missing_keys), list(unexpected_keys)


def build_model_from_checkpoint(
    family: str,
    checkpoint_path: str | Path,
    num_classes: int = 21,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[nn.Module, list[str], list[str]]:
    model = build_model(family=family, num_classes=num_classes)
    missing_keys, unexpected_keys = load_checkpoint(
        model,
        checkpoint_path=checkpoint_path,
        map_location=map_location,
        strict=strict,
    )
    return model, missing_keys, unexpected_keys
