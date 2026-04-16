from __future__ import annotations

import torch

from src.attacks.base import AttackOutput, SegmentationAttack
from src.attacks.constraints import project_linf
from src.attacks.losses import (
    build_balanced_class_weights,
    per_image_segmentation_accuracy,
    segmentation_js_divergence_loss,
    segmentation_masked_cross_entropy_balanced_loss,
    segmentation_masked_cross_entropy_loss,
)


class SEAAttack(SegmentationAttack):
    """Sequential ensemble attack from Robust-Segmentation."""

    attack_name = "sea"
    phase_names = ("mask-ce-bal", "mask-ce", "js-avg")

    def _compute_phase_loss(
        self,
        phase_name: str,
        logits: torch.Tensor,
        targets: torch.Tensor,
        balanced_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if phase_name == "mask-ce-bal":
            return segmentation_masked_cross_entropy_balanced_loss(
                logits=logits,
                targets=targets,
                ignore_index=self.config.ignore_index,
                class_weights=balanced_weights,
            )
        if phase_name == "mask-ce":
            return segmentation_masked_cross_entropy_loss(
                logits=logits,
                targets=targets,
                ignore_index=self.config.ignore_index,
            )
        if phase_name == "js-avg":
            return segmentation_js_divergence_loss(
                logits=logits,
                targets=targets,
                ignore_index=self.config.ignore_index,
            )
        raise KeyError(f"Unsupported SEA phase: {phase_name}")

    def _phase_objective(
        self,
        attack_input: torch.Tensor,
        *,
        phase_name: str,
        targets: torch.Tensor,
        balanced_weights: torch.Tensor | None,
        direction: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = self.model.logits(attack_input)
        loss = self._compute_phase_loss(
            phase_name=phase_name,
            logits=logits,
            targets=targets,
            balanced_weights=balanced_weights,
        )
        return direction * loss, {"loss": float(loss.detach().cpu().item())}

    def _run_phase(
        self,
        clean: torch.Tensor,
        labels: torch.Tensor,
        phase_name: str,
        start: torch.Tensor,
        balanced_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, float]:
        epsilon = self.config.epsilon
        step_size = self.config.resolved_step_size()
        steps = self.config.steps
        direction = -1.0 if self.config.targeted else 1.0

        adversarial = project_linf(
            adversarial_images=start,
            clean_images=clean,
            epsilon=epsilon,
            min_value=self.config.clamp_min,
            max_value=self.config.clamp_max,
        )
        loss_value = 0.0

        with torch.enable_grad():
            for _ in range(steps):
                gradient, stats = self.estimate_input_gradient(
                    adversarial,
                    lambda attack_input: self._phase_objective(
                        attack_input=attack_input,
                        phase_name=phase_name,
                        targets=labels,
                        balanced_weights=balanced_weights,
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

        return adversarial.detach(), loss_value

    def run(self, images: torch.Tensor, targets: torch.Tensor) -> AttackOutput:
        clean = images.detach().clone().to(self.model.device)
        labels = targets.detach().clone().to(self.model.device)
        current = clean.clone()

        if self.config.random_start:
            current = clean + torch.empty_like(clean).uniform_(-self.config.epsilon, self.config.epsilon)
            current = project_linf(
                adversarial_images=current,
                clean_images=clean,
                epsilon=self.config.epsilon,
                min_value=self.config.clamp_min,
                max_value=self.config.clamp_max,
            )

        balanced_weights = build_balanced_class_weights(
            targets=labels,
            num_classes=self.model.num_classes,
            ignore_index=self.config.ignore_index,
        )
        best_adversarial = current.clone()
        with torch.no_grad():
            best_score = per_image_segmentation_accuracy(
                logits=self.model.logits(current),
                targets=labels,
                ignore_index=self.config.ignore_index,
            )

        phase_metrics: list[dict[str, float | str]] = []
        for phase_name in self.phase_names:
            current, phase_loss = self._run_phase(
                clean=clean,
                labels=labels,
                phase_name=phase_name,
                start=current,
                balanced_weights=balanced_weights,
            )
            with torch.no_grad():
                phase_logits = self.model.logits(current)
                phase_score = per_image_segmentation_accuracy(
                    logits=phase_logits,
                    targets=labels,
                    ignore_index=self.config.ignore_index,
                )

            if self.config.targeted:
                update_mask = phase_score >= best_score
                best_score = torch.maximum(best_score, phase_score)
            else:
                update_mask = phase_score <= best_score
                best_score = torch.minimum(best_score, phase_score)
            best_adversarial[update_mask] = current[update_mask]
            phase_metrics.append(
                {
                    "phase": phase_name,
                    "loss": phase_loss,
                    "mean_score": float(phase_score.mean().cpu().item()),
                }
            )

        perturbation = best_adversarial - clean
        return AttackOutput(
            adversarial_images=best_adversarial.detach(),
            perturbation=perturbation.detach(),
            metadata={
                "attack": self.attack_name,
                "steps": self.config.steps,
                "step_size": self.config.resolved_step_size(),
                "epsilon": self.config.epsilon,
                "random_start": self.config.random_start,
                "targeted": self.config.targeted,
                "phases": phase_metrics,
                "best_mean_score": float(best_score.mean().cpu().item()),
            },
        )
