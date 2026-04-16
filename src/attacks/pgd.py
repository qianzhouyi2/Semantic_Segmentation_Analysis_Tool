from __future__ import annotations

import torch

from src.attacks.base import AttackOutput, SegmentationAttack
from src.attacks.constraints import project_linf
from src.attacks.losses import segmentation_attack_loss


class PGDAttack(SegmentationAttack):
    """Multi-step Linf PGD for dense segmentation models."""

    attack_name = "pgd"

    def compute_attack_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        step_index: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step_index, total_steps
        return segmentation_attack_loss(
            logits=logits,
            targets=targets,
            loss_name=self.config.loss_name,
            ignore_index=self.config.ignore_index,
        )

    def _objective(
        self,
        attack_input: torch.Tensor,
        targets: torch.Tensor,
        step_index: int,
        total_steps: int,
        direction: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = self.model.logits(attack_input)
        loss = self.compute_attack_loss(
            logits=logits,
            targets=targets,
            step_index=step_index,
            total_steps=total_steps,
        )
        return direction * loss, {"loss": float(loss.detach().cpu().item())}

    def run(self, images: torch.Tensor, targets: torch.Tensor) -> AttackOutput:
        epsilon = self.config.epsilon
        step_size = self.config.resolved_step_size()
        steps = self.config.steps

        clean = images.detach().clone().to(self.model.device)
        labels = targets.detach().clone().to(self.model.device)
        adversarial = clean.clone()

        if self.config.random_start:
            adversarial = clean + torch.empty_like(clean).uniform_(-epsilon, epsilon)
            adversarial = project_linf(
                adversarial_images=adversarial,
                clean_images=clean,
                epsilon=epsilon,
                min_value=self.config.clamp_min,
                max_value=self.config.clamp_max,
            )

        loss_value = 0.0
        direction = -1.0 if self.config.targeted else 1.0
        with torch.enable_grad():
            for step_index in range(steps):
                gradient, stats = self.estimate_input_gradient(
                    adversarial,
                    lambda attack_input: self._objective(
                        attack_input=attack_input,
                        targets=labels,
                        step_index=step_index,
                        total_steps=steps,
                        direction=direction,
                    ),
                )
                adversarial = adversarial.detach() + step_size * gradient.sign()
                adversarial = project_linf(
                    adversarial_images=adversarial,
                    clean_images=clean,
                    epsilon=epsilon,
                    min_value=self.config.clamp_min,
                    max_value=self.config.clamp_max,
                )
                loss_value = float(stats["loss"]) if isinstance(stats, dict) else float(stats)

        perturbation = adversarial - clean
        return AttackOutput(
            adversarial_images=adversarial.detach(),
            perturbation=perturbation.detach(),
            metadata={
                "attack": self.attack_name,
                "steps": steps,
                "step_size": step_size,
                "loss": loss_value,
                "random_start": self.config.random_start,
                "targeted": self.config.targeted,
            },
        )
