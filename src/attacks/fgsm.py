from __future__ import annotations

import torch

from src.attacks.base import AttackOutput, SegmentationAttack
from src.attacks.constraints import clamp_images
from src.attacks.losses import segmentation_attack_loss


class FGSMAttack(SegmentationAttack):
    """Single-step attack placeholder with a usable default implementation."""

    def run(self, images: torch.Tensor, targets: torch.Tensor) -> AttackOutput:
        epsilon = self.config.epsilon
        if epsilon <= 0:
            raise ValueError("FGSM epsilon must be positive.")

        inputs = images.detach().clone().to(self.model.device)
        labels = targets.detach().clone().to(self.model.device)
        inputs.requires_grad_(True)

        logits = self.model.logits(inputs)
        loss = segmentation_attack_loss(
            logits=logits,
            targets=labels,
            loss_name=self.config.loss_name,
            ignore_index=self.config.ignore_index,
        )
        if self.config.targeted:
            loss = -loss
        loss.backward()

        step = epsilon * inputs.grad.sign()
        adversarial = inputs + step
        adversarial = clamp_images(adversarial, self.config.clamp_min, self.config.clamp_max).detach()
        perturbation = adversarial - images.to(self.model.device)
        return AttackOutput(
            adversarial_images=adversarial,
            perturbation=perturbation,
            metadata={"loss": float(loss.detach().cpu().item()), "attack": "fgsm"},
        )
