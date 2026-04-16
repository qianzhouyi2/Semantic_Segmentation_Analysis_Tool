from __future__ import annotations

import torch

from src.attacks.base import AttackOutput, SegmentationAttack
from src.attacks.constraints import project_linf
from src.attacks.losses import (
    build_safe_targets,
    segmentation_attack_loss_map,
    segmentation_kl_divergence_map,
    spatial_mean,
)


class TranSegPGDAttack(SegmentationAttack):
    """Two-stage TranSegPGD adapted from the paper formulation."""

    attack_name = "transegpgd"

    def gamma(self) -> float:
        return float(self.config.extra.get("gamma", 0.1))

    def beta(self) -> float:
        return float(self.config.extra.get("beta", 0.1))

    def _objective(
        self,
        attack_input: torch.Tensor,
        *,
        labels: torch.Tensor,
        clean_logits: torch.Tensor,
        direction: float,
    ) -> tuple[torch.Tensor, dict[str, float | str]]:
        logits = self.model.logits(attack_input)
        loss_map, valid_mask = segmentation_attack_loss_map(
            logits=logits,
            targets=labels,
            loss_name=self.config.loss_name,
            ignore_index=self.config.ignore_index,
        )
        if not valid_mask.any():
            return logits.sum() * 0.0, {"loss": 0.0, "last_stage": "stage1"}

        safe_targets = build_safe_targets(targets=labels, valid_mask=valid_mask)
        predictions = logits.argmax(dim=1)
        correct_mask = valid_mask & (predictions == safe_targets)
        incorrect_mask = valid_mask & ~correct_mask

        if correct_mask.any():
            weighted_map = (
                (1.0 - self.gamma()) * correct_mask.float().detach()
                + self.gamma() * incorrect_mask.float().detach()
            ) * loss_map
            stage_name = "stage1"
        else:
            kl_map, kl_valid_mask = segmentation_kl_divergence_map(
                adv_logits=logits,
                clean_logits=clean_logits,
                targets=labels,
                ignore_index=self.config.ignore_index,
            )
            mean_kl = kl_map.masked_select(kl_valid_mask).mean()
            high_transfer_mask = valid_mask & (kl_map > mean_kl)
            low_transfer_mask = valid_mask & ~high_transfer_mask
            weighted_map = (
                (1.0 - self.beta()) * high_transfer_mask.float().detach()
                + self.beta() * low_transfer_mask.float().detach()
            ) * loss_map
            stage_name = "stage2"

        loss = spatial_mean(weighted_map).mean()
        return direction * loss, {
            "loss": float(loss.detach().cpu().item()),
            "last_stage": stage_name,
        }

    def run(self, images: torch.Tensor, targets: torch.Tensor) -> AttackOutput:
        epsilon = self.config.epsilon
        step_size = self.config.resolved_step_size()
        steps = self.config.steps
        direction = -1.0 if self.config.targeted else 1.0

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

        with torch.no_grad():
            clean_logits = self.model.logits(clean)

        last_stage = "stage1"
        stage1_steps = 0
        stage2_steps = 0
        loss_value = 0.0
        with torch.enable_grad():
            for _ in range(steps):
                gradient, stats = self.estimate_input_gradient(
                    adversarial,
                    lambda attack_input: self._objective(
                        attack_input=attack_input,
                        labels=labels,
                        clean_logits=clean_logits,
                        direction=direction,
                    ),
                )
                if isinstance(stats, dict):
                    last_stage = str(stats["last_stage"])
                    loss_value = float(stats["loss"])
                else:
                    loss_value = float(stats)
                if last_stage == "stage1":
                    stage1_steps += 1
                else:
                    stage2_steps += 1
                adversarial = adversarial.detach() + step_size * gradient.sign()
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
            metadata={
                "attack": self.attack_name,
                "steps": steps,
                "step_size": step_size,
                "epsilon": epsilon,
                "loss": loss_value,
                "last_stage": last_stage,
                "stage1_steps": stage1_steps,
                "stage2_steps": stage2_steps,
                "gamma": self.gamma(),
                "beta": self.beta(),
                "random_start": self.config.random_start,
                "targeted": self.config.targeted,
            },
        )
