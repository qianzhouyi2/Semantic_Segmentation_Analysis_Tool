from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any

import torch

from src.models.base import SegmentationModelAdapter
from src.models.sparse import SPARSE_ATTACK_BACKWARD_MODE_CHOICES


ATTACK_BACKWARD_MODE_CHOICES = SPARSE_ATTACK_BACKWARD_MODE_CHOICES
RESTART_SELECTION_CRITERION = "per_image_accuracy"


def _average_auxiliary_values(auxiliary_values: list[Any]) -> Any:
    if not auxiliary_values:
        return None

    first_value = auxiliary_values[0]
    if first_value is None:
        return None

    if torch.is_tensor(first_value):
        if first_value.ndim == 0:
            return sum(float(value.detach().cpu().item()) for value in auxiliary_values) / float(len(auxiliary_values))
        return first_value.detach()

    if isinstance(first_value, (int, float)):
        return sum(float(value) for value in auxiliary_values) / float(len(auxiliary_values))

    if isinstance(first_value, Mapping):
        aggregated: dict[str, Any] = {}
        seen_keys: list[str] = []
        for item in auxiliary_values:
            if not isinstance(item, Mapping):
                continue
            for key in item:
                if key not in seen_keys:
                    seen_keys.append(key)

        for key in seen_keys:
            values = [item[key] for item in auxiliary_values if isinstance(item, Mapping) and key in item]
            if not values:
                continue
            value = values[0]
            if torch.is_tensor(value) and value.ndim == 0:
                aggregated[key] = sum(float(entry.detach().cpu().item()) for entry in values) / float(len(values))
            elif isinstance(value, (int, float)):
                aggregated[key] = sum(float(entry) for entry in values) / float(len(values))
            else:
                aggregated[key] = values[-1]
        return aggregated

    return auxiliary_values[-1]


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
    attack_backward_mode: str = "default"
    num_restarts: int = 1
    eot_iters: int = 1
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
        self.attack_backward_mode = str(self.attack_backward_mode).strip().lower()
        self.num_restarts = int(self.num_restarts)
        self.eot_iters = int(self.eot_iters)
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
        if self.attack_backward_mode not in ATTACK_BACKWARD_MODE_CHOICES:
            raise ValueError(
                f"Unsupported attack_backward_mode `{self.attack_backward_mode}`. "
                f"Expected one of {ATTACK_BACKWARD_MODE_CHOICES}."
            )
        if self.num_restarts < 1:
            raise ValueError("Attack num_restarts must be at least 1.")
        if self.eot_iters < 1:
            raise ValueError("Attack eot_iters must be at least 1.")

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

    def scaled(self, epsilon_scale: float) -> "AttackConfig":
        if epsilon_scale <= 0:
            raise ValueError("epsilon_scale must be positive.")

        extra = dict(self.extra)
        extra["epsilon_scale"] = float(epsilon_scale)
        extra["base_epsilon"] = self.epsilon
        if self.step_size is not None:
            extra["base_step_size"] = self.step_size

        return replace(
            self,
            epsilon=self.epsilon * float(epsilon_scale),
            step_size=None if self.step_size is None else self.step_size * float(epsilon_scale),
            extra=extra,
        )

    def with_radius_255(self, radius_255: float) -> "AttackConfig":
        if radius_255 < 0 or radius_255 > 255:
            raise ValueError("radius_255 must be within [0, 255].")

        radius = float(radius_255)
        epsilon = radius / 255.0
        if self.step_size is None:
            step_size = None
        elif self.epsilon > 0:
            step_size = self.step_size * (epsilon / self.epsilon)
        else:
            step_size = 0.0

        extra = dict(self.extra)
        extra["epsilon_radius_255"] = radius
        extra["base_epsilon"] = self.epsilon
        if self.step_size is not None:
            extra["base_step_size"] = self.step_size
        extra["epsilon_scale"] = 0.0 if self.epsilon == 0 else (epsilon / self.epsilon)

        return replace(
            self,
            epsilon=epsilon,
            step_size=step_size,
            extra=extra,
        )

    def epsilon_radius_255(self) -> float | None:
        raw_value = self.extra.get("epsilon_radius_255")
        return None if raw_value is None else float(raw_value)

    def with_runtime_overrides(
        self,
        *,
        epsilon_scale: float | None = None,
        epsilon_radius_255: float | None = None,
        attack_backward_mode: str | None = None,
        num_restarts: int | None = None,
        eot_iters: int | None = None,
    ) -> "AttackConfig":
        config = self
        if epsilon_radius_255 is not None:
            config = config.with_radius_255(epsilon_radius_255)
        elif epsilon_scale is not None and float(epsilon_scale) != 1.0:
            config = config.scaled(float(epsilon_scale))

        return replace(
            config,
            attack_backward_mode=config.attack_backward_mode if attack_backward_mode is None else attack_backward_mode,
            num_restarts=config.num_restarts if num_restarts is None else num_restarts,
            eot_iters=config.eot_iters if eot_iters is None else eot_iters,
        )

    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "attack_backward_mode": self.attack_backward_mode,
            "num_restarts": self.num_restarts,
            "eot_iters": self.eot_iters,
            "epsilon_radius_255": self.epsilon_radius_255(),
            "effective_epsilon_scale": float(self.extra.get("epsilon_scale", 1.0)),
            "sample_wise_worst_case_over_restarts": self.num_restarts > 1,
            "restart_selection": RESTART_SELECTION_CRITERION if self.num_restarts > 1 else None,
        }


@dataclass(slots=True)
class AttackOutput:
    adversarial_images: torch.Tensor
    perturbation: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class SegmentationAttack(ABC):
    def __init__(self, model: SegmentationModelAdapter, config: AttackConfig) -> None:
        self.model = model
        self.config = config

    def eot_iters(self) -> int:
        return self.config.eot_iters

    def estimate_input_gradient(
        self,
        inputs: torch.Tensor,
        objective_fn: Callable[[torch.Tensor], tuple[torch.Tensor, Any]],
    ) -> tuple[torch.Tensor, Any]:
        gradient_sum = torch.zeros_like(inputs)
        auxiliary_values: list[Any] = []

        for _ in range(self.eot_iters()):
            attack_input = inputs.detach().requires_grad_(True)
            objective, auxiliary = objective_fn(attack_input)
            gradient = torch.autograd.grad(objective, attack_input, retain_graph=False, create_graph=False)[0]
            gradient_sum += gradient.detach()
            auxiliary_values.append(auxiliary)

        return gradient_sum / float(self.eot_iters()), _average_auxiliary_values(auxiliary_values)

    @abstractmethod
    def run(self, images: torch.Tensor, targets: torch.Tensor) -> AttackOutput:
        raise NotImplementedError
