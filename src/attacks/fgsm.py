from __future__ import annotations

import torch

from src.attacks.base import AttackOutput, SegmentationAttack
from src.attacks.constraints import clamp_images
from src.attacks.losses import segmentation_attack_loss_map


class FGSMAttack(SegmentationAttack):
    """Single-step Linf FGSM following masked per-pixel cross-entropy."""

    def _objective(self, attack_input: torch.Tensor, targets: torch.Tensor, multiplier: float) -> tuple[torch.Tensor, dict[str, float | int]]:
        logits = self.model.logits(attack_input)
        loss_map, valid_mask = segmentation_attack_loss_map(
            logits=logits,
            targets=targets,
            loss_name=self.config.loss_name,
            ignore_index=self.config.ignore_index,
        )
        objective = multiplier * loss_map.masked_select(valid_mask).sum()
        valid_pixels = int(valid_mask.sum().detach().cpu().item())
        if valid_pixels == 0:
            loss_value = 0.0
        else:
            loss_value = float(loss_map.masked_select(valid_mask).mean().detach().cpu().item())
        return objective, {
            "loss": loss_value,
            "valid_pixels": valid_pixels,
        }

    def run(self, images: torch.Tensor, targets: torch.Tensor) -> AttackOutput:
        epsilon = self.config.epsilon
        if epsilon <= 0:
            raise ValueError("FGSM epsilon must be positive.")

        inputs = images.detach().clone().to(self.model.device)
        labels = targets.detach().clone().to(self.model.device)
        multiplier = -1.0 if self.config.targeted else 1.0
        with torch.enable_grad():
            gradient, stats = self.estimate_input_gradient(
                inputs,
                lambda attack_input: self._objective(
                    attack_input=attack_input,
                    targets=labels,
                    multiplier=multiplier,
                ),
            )

        inputs = inputs.detach()
        step = epsilon * gradient.sign()
        adversarial = inputs + step
        adversarial = clamp_images(adversarial, self.config.clamp_min, self.config.clamp_max).detach()
        perturbation = adversarial - images.to(self.model.device)
        valid_pixels = int(stats["valid_pixels"]) if isinstance(stats, dict) else 0
        loss_value = float(stats["loss"]) if isinstance(stats, dict) else 0.0
        return AttackOutput(
            adversarial_images=adversarial,
            perturbation=perturbation,
            metadata={
                "attack": "fgsm",
                "loss": loss_value,
                "targeted": self.config.targeted,
                "valid_pixels": valid_pixels,
            },
        )
