from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
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

    def __post_init__(self) -> None:
        self.name = str(self.name).strip().lower()
        self.epsilon = float(self.epsilon)
        self.step_size = None if self.step_size is None else float(self.step_size)
        self.steps = int(self.steps)
        self.clamp_min = float(self.clamp_min)
        self.clamp_max = float(self.clamp_max)
        self.targeted = bool(self.targeted)
        self.random_start = bool(self.random_start)
        self.loss_name = str(self.loss_name).strip().lower()
        self.ignore_index = None if self.ignore_index is None else int(self.ignore_index)
        self.extra = dict(self.extra)

        if not self.name:
            raise ValueError("Attack name must be a non-empty string.")
        if self.epsilon < 0:
            raise ValueError("Attack epsilon must be non-negative.")
        if self.step_size is not None and self.step_size < 0:
            raise ValueError("Attack step_size must be non-negative when provided.")
        if self.steps < 1:
            raise ValueError("Attack steps must be at least 1.")
        if self.clamp_min > self.clamp_max:
            raise ValueError("Attack clamp_min must be less than or equal to clamp_max.")

    @classmethod
    def from_dict(cls, raw_config: Mapping[str, Any]) -> "AttackConfig":
        field_names = {entry.name for entry in fields(cls) if entry.name != "extra"}
        payload = {key: raw_config[key] for key in field_names if key in raw_config}
        extra = raw_config.get("extra", {})
        if extra is None:
            extra = {}
        if not isinstance(extra, Mapping):
            raise ValueError("Attack config `extra` must be a mapping when provided.")
        payload["extra"] = dict(extra)
        for key, value in raw_config.items():
            if key not in field_names and key != "extra":
                payload["extra"][key] = value
        return cls(**payload)

    def resolved_step_size(self) -> float:
        if self.step_size is not None:
            return self.step_size
        if self.steps == 0:
            return 0.0
        return self.epsilon / float(self.steps)


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
