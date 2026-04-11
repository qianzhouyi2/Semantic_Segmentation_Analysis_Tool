from __future__ import annotations

import torch
import torch.nn.functional as F


def segmentation_attack_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str = "cross_entropy",
    ignore_index: int | None = None,
) -> torch.Tensor:
    if loss_name != "cross_entropy":
        raise NotImplementedError(f"Unsupported attack loss: {loss_name}")

    kwargs = {}
    if ignore_index is not None:
        kwargs["ignore_index"] = ignore_index
    return F.cross_entropy(logits, targets.long(), **kwargs)
