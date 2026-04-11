from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from src.models.base import SegmentationModelAdapter


@dataclass(slots=True)
class AttackConfig:
    name: str
    epsilon: float
    step_size: float | None = None
    steps: int = 1
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    random_start: bool = False
    targeted: bool = False
    loss_name: str = "cross_entropy"
    ignore_index: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AttackOutput:
    adversarial_images: torch.Tensor
    perturbation: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class SegmentationAttack(ABC):
    def __init__(self, model: SegmentationModelAdapter, config: AttackConfig) -> None:
        self.model = model
        self.config = config

    @abstractmethod
    def run(self, images: torch.Tensor, targets: torch.Tensor) -> AttackOutput:
        raise NotImplementedError
