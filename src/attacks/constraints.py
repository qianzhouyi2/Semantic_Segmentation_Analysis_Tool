from __future__ import annotations

import torch


def clamp_images(images: torch.Tensor, min_value: float = 0.0, max_value: float = 1.0) -> torch.Tensor:
    return images.clamp(min_value, max_value)


def project_linf(
    adversarial_images: torch.Tensor,
    clean_images: torch.Tensor,
    epsilon: float,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> torch.Tensor:
    perturbation = (adversarial_images - clean_images).clamp(-epsilon, epsilon)
    return (clean_images + perturbation).clamp(min_value, max_value)
