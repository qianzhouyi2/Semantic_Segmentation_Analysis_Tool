"""Model adapters for segmentation inference and attacks."""

from src.models.build import MODEL_FAMILY_CHOICES, build_model, build_model_from_checkpoint, load_checkpoint
from src.models.base import SegmentationModelAdapter, TorchSegmentationModelAdapter
from src.models.registry import create_model_adapter, register_model_adapter

__all__ = [
    "MODEL_FAMILY_CHOICES",
    "SegmentationModelAdapter",
    "TorchSegmentationModelAdapter",
    "build_model",
    "build_model_from_checkpoint",
    "create_model_adapter",
    "load_checkpoint",
    "register_model_adapter",
]
