from __future__ import annotations

import torch

from src.attacks.base import AttackOutput, SegmentationAttack
from src.attacks.constraints import clamp_images, project_linf
from src.attacks.losses import segmentation_attack_loss


class PGDAttack(SegmentationAttack):
    """Multi-step L-infinity PGD scaffold for future robustness experiments."""

    def run(self, images: torch.Tensor, targets: torch.Tensor) -> AttackOutput:
        epsilon = self.config.epsilon
        step_size = self.config.step_size or (epsilon / max(self.config.steps, 1))
        steps = max(int(self.config.steps), 1)

        clean = images.detach().clone().to(self.model.device)
        labels = targets.detach().clone().to(self.model.device)
        adversarial = clean.clone()

        if self.config.random_start:
            adversarial = adversarial + torch.empty_like(adversarial).uniform_(-epsilon, epsilon)
            adversarial = clamp_images(adversarial, self.config.clamp_min, self.config.clamp_max)

        for _ in range(steps):
            adversarial.requires_grad_(True)
            logits = self.model.logits(adversarial)
            loss = segmentation_attack_loss(
                logits=logits,
                targets=labels,
                loss_name=self.config.loss_name,
                ignore_index=self.config.ignore_index,
            )
            if self.config.targeted:
                loss = -loss
            loss.backward()

            step = step_size * adversarial.grad.sign()
            adversarial = adversarial.detach() + step
            adversarial = project_linf(
                adversarial_images=adversarial,
                clean_images=clean,
                epsilon=epsilon,
                min_value=self.config.clamp_min,
                max_value=self.config.clamp_max,
            )

        perturbation = adversarial - clean
        return AttackOutput(
            adversarial_images=adversarial.detach(),
            perturbation=perturbation.detach(),
            metadata={"steps": steps, "step_size": step_size, "attack": "pgd"},
        )
