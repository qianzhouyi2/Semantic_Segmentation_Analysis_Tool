from __future__ import annotations

import torch

from src.attacks.losses import segmentation_segpgd_loss
from src.attacks.pgd import PGDAttack


class SegPGDAttack(PGDAttack):
    """SegPGD for semantic segmentation with dynamic weighting on hard pixels."""

    attack_name = "segpgd"

    def compute_attack_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        step_index: int,
        total_steps: int,
    ) -> torch.Tensor:
        return segmentation_segpgd_loss(
            logits=logits,
            targets=targets,
            iteration=step_index + 1,
            iterations=total_steps,
            loss_name=self.config.loss_name,
            ignore_index=self.config.ignore_index,
            targeted=self.config.targeted,
        )
