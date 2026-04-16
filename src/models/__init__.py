"""Model adapters for segmentation inference and attacks."""

from src.models.build import MODEL_FAMILY_CHOICES, build_model, build_model_from_checkpoint, load_checkpoint
from src.models.base import SegmentationModelAdapter, TorchSegmentationModelAdapter
from src.models.registry import create_model_adapter, register_model_adapter
from src.models.sparse import SPARSE_DEFENSE_CHOICES, SparseDefenseConfig, load_sparse_defense_config

__all__ = [
    "MODEL_FAMILY_CHOICES",
    "SPARSE_DEFENSE_CHOICES",
    "SegmentationModelAdapter",
    "SparseDefenseConfig",
    "TorchSegmentationModelAdapter",
    "build_model",
    "build_model_from_checkpoint",
    "create_model_adapter",
    "load_checkpoint",
    "load_sparse_defense_config",
    "register_model_adapter",
]
