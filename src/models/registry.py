from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.models.base import SegmentationModelAdapter


MODEL_ADAPTERS: dict[str, Callable[..., SegmentationModelAdapter]] = {}


def register_model_adapter(name: str, factory: Callable[..., SegmentationModelAdapter]) -> None:
    MODEL_ADAPTERS[name] = factory


def create_model_adapter(name: str, **kwargs: Any) -> SegmentationModelAdapter:
    if name not in MODEL_ADAPTERS:
        available = ", ".join(sorted(MODEL_ADAPTERS)) or "<none>"
        raise KeyError(f"Unknown model adapter '{name}'. Available: {available}")
    return MODEL_ADAPTERS[name](**kwargs)
