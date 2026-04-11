from __future__ import annotations

import torch

from src.attacks.base import AttackOutput, SegmentationAttack
from src.attacks.constraints import project_linf
from src.attacks.losses import build_safe_targets, build_valid_mask, sample_alternate_labels


class DAGAttack(SegmentationAttack):
    """Dense Adversary Generation adapted to the current segmentation pipeline."""

    attack_name = "dag"

    def run(self, images: torch.Tensor, targets: torch.Tensor) -> AttackOutput:
        if self.config.targeted:
            raise NotImplementedError("DAG targeted mode is not supported in this pipeline.")

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

        num_classes = self.model.num_classes
        target_labels = sample_alternate_labels(
            targets=labels,
            num_classes=num_classes,
            ignore_index=self.config.ignore_index,
        )
        valid_mask = build_valid_mask(targets=labels, num_classes=num_classes, ignore_index=self.config.ignore_index)
        safe_targets = build_safe_targets(targets=labels, valid_mask=valid_mask)
        safe_target_labels = torch.where(valid_mask, target_labels, torch.zeros_like(target_labels))

        active_pixels = int(valid_mask.sum().detach().cpu().item())
        loss_value = 0.0
        with torch.enable_grad():
            for _ in range(steps):
                adversarial = adversarial.detach().requires_grad_(True)
                logits = self.model.logits(adversarial)
                predictions = logits.argmax(dim=1)
                active_mask = valid_mask & (predictions == safe_targets)
                active_pixels = int(active_mask.sum().detach().cpu().item())
                if active_pixels == 0:
                    adversarial = adversarial.detach()
                    break

                true_logits = logits.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
                target_logits = logits.gather(1, safe_target_labels.unsqueeze(1)).squeeze(1)
                objective_map = active_mask.float().detach() * (target_logits - true_logits)
                loss = objective_map.sum() / active_mask.float().sum().clamp_min(1.0)
                gradient = torch.autograd.grad(loss, adversarial, retain_graph=False, create_graph=False)[0]
                adversarial = adversarial.detach() + step_size * gradient.sign()
                adversarial = project_linf(
                    adversarial_images=adversarial,
                    clean_images=clean,
                    epsilon=epsilon,
                    min_value=self.config.clamp_min,
                    max_value=self.config.clamp_max,
                )
                loss_value = float(loss.detach().cpu().item())

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
                "active_pixels": active_pixels,
                "random_start": self.config.random_start,
            },
        )
