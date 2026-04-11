from __future__ import annotations

import torch

from src.attacks.losses import segmentation_cospgd_loss
from src.attacks.pgd import PGDAttack


class CosPGDAttack(PGDAttack):
    """Linf CosPGD for semantic segmentation using cosine-scaled pixel losses."""

    attack_name = "cospgd"

    def compute_attack_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        step_index: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step_index, total_steps
        return segmentation_cospgd_loss(
            logits=logits,
            targets=targets,
            loss_name=self.config.loss_name,
            ignore_index=self.config.ignore_index,
            targeted=self.config.targeted,
        )
