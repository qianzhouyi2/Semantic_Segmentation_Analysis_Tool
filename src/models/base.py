from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ModelBatch:
    images: torch.Tensor
    masks: torch.Tensor | None = None


class SegmentationModelAdapter(ABC):
    """Minimal adapter interface used by adversarial attacks."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def logits(self, images: torch.Tensor) -> torch.Tensor:
        """Return dense segmentation logits with shape [B, C, H, W]."""
        raise NotImplementedError

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.logits(images).argmax(dim=1)
