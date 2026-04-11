from __future__ import annotations

import numpy as np


def normalize_perturbation(perturbation: np.ndarray) -> np.ndarray:
    """Convert a perturbation tensor/image into a displayable 0-255 map."""
    if perturbation.size == 0:
        return perturbation.astype(np.uint8)

    data = perturbation.astype(np.float32)
    min_value = float(data.min())
    max_value = float(data.max())
    if max_value == min_value:
        return np.zeros_like(data, dtype=np.uint8)
    normalized = (data - min_value) / (max_value - min_value)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)
