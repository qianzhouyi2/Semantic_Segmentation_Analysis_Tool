from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def normalize_gradient_by_mean_abs(gradient: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    scale = gradient.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    return gradient / scale


def input_diversity(
    images: torch.Tensor,
    resize_rate: float = 0.9,
    diversity_prob: float = 0.7,
    pad_value: float = 0.0,
) -> torch.Tensor:
    if diversity_prob <= 0.0 or torch.rand(1, device=images.device).item() > diversity_prob:
        return images

    height, width = images.shape[-2:]
    min_height = max(1, int(round(height * resize_rate)))
    min_width = max(1, int(round(width * resize_rate)))
    if min_height >= height and min_width >= width:
        return images

    rnd_height = int(torch.randint(min_height, height + 1, (1,), device=images.device).item())
    rnd_width = int(torch.randint(min_width, width + 1, (1,), device=images.device).item())
    resized = F.interpolate(images, size=(rnd_height, rnd_width), mode="bilinear", align_corners=False)

    pad_height = height - rnd_height
    pad_width = width - rnd_width
    pad_top = int(torch.randint(0, pad_height + 1, (1,), device=images.device).item())
    pad_left = int(torch.randint(0, pad_width + 1, (1,), device=images.device).item())
    pad_bottom = pad_height - pad_top
    pad_right = pad_width - pad_left
    return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)


def build_gaussian_kernel(
    channels: int,
    kernel_size: int = 5,
    sigma: float = 3.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("Gaussian kernel size must be odd.")

    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    if dtype is not None:
        kernel = kernel.to(dtype=dtype)
    return kernel


def smooth_translation_invariant_gradient(
    gradient: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 3.0,
) -> torch.Tensor:
    kernel = build_gaussian_kernel(
        channels=gradient.size(1),
        kernel_size=kernel_size,
        sigma=sigma,
        device=gradient.device,
        dtype=gradient.dtype,
    )
    padding = kernel_size // 2
    return F.conv2d(gradient, kernel, padding=padding, groups=gradient.size(1))


def cosine_threshold(cosine_bins: float) -> float:
    return math.cos(math.pi / max(float(cosine_bins), 1.0))


def select_feature_map(
    features: dict[str, torch.Tensor],
    preferred_key: str | None = None,
) -> torch.Tensor:
    if preferred_key and preferred_key in features:
        return features[preferred_key]

    for key in ("backbone:last", "encoder", "logits", "backbone:first"):
        if key in features:
            return features[key]

    if not features:
        raise KeyError("No feature maps were returned by the model adapter.")
    return next(iter(features.values()))
