from __future__ import annotations

import torch

from src.attacks.base import AttackOutput, SegmentationAttack
from src.attacks.constraints import project_linf
from src.attacks.losses import segmentation_attack_loss
from src.attacks.utils import (
    input_diversity,
    normalize_gradient_by_mean_abs,
    smooth_translation_invariant_gradient,
)


class TransferIterativeAttack(SegmentationAttack):
    """Shared Linf transfer-attack loop for segmentation models."""

    attack_name = "transfer"
    use_momentum = False
    use_nesterov = False
    use_diversity = False
    use_ti = False

    def momentum_decay(self) -> float:
        return float(self.config.extra.get("momentum_decay", 1.0))

    def diversity_prob(self) -> float:
        return float(self.config.extra.get("diversity_prob", 0.7))

    def resize_rate(self) -> float:
        return float(self.config.extra.get("resize_rate", 0.9))

    def ti_kernel_size(self) -> int:
        return int(self.config.extra.get("ti_kernel_size", 5))

    def ti_sigma(self) -> float:
        return float(self.config.extra.get("ti_sigma", 3.0))

    def compute_attack_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return segmentation_attack_loss(
            logits=logits,
            targets=targets,
            loss_name=self.config.loss_name,
            ignore_index=self.config.ignore_index,
        )

    def postprocess_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.use_ti:
            gradient = smooth_translation_invariant_gradient(
                gradient=gradient,
                kernel_size=self.ti_kernel_size(),
                sigma=self.ti_sigma(),
            )
        return gradient

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

        momentum = torch.zeros_like(clean)
        loss_value = 0.0
        with torch.enable_grad():
            for _ in range(steps):
                attack_input = adversarial
                if self.use_nesterov:
                    attack_input = attack_input + step_size * self.momentum_decay() * momentum
                attack_input = attack_input.detach().requires_grad_(True)
                model_input = attack_input
                if self.use_diversity:
                    model_input = input_diversity(
                        images=attack_input,
                        resize_rate=self.resize_rate(),
                        diversity_prob=self.diversity_prob(),
                        pad_value=float(self.config.clamp_min),
                    )

                logits = self.model.logits(model_input)
                loss = self.compute_attack_loss(logits=logits, targets=labels)
                objective = direction * loss
                gradient = torch.autograd.grad(objective, attack_input, retain_graph=False, create_graph=False)[0]
                gradient = self.postprocess_gradient(gradient)
                if self.use_momentum:
                    gradient = normalize_gradient_by_mean_abs(gradient)
                    momentum = self.momentum_decay() * momentum + gradient
                    update = momentum
                else:
                    update = gradient

                adversarial = adversarial.detach() + step_size * update.sign()
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
                "random_start": self.config.random_start,
                "targeted": self.config.targeted,
                "momentum": self.use_momentum,
                "nesterov": self.use_nesterov,
                "diversity": self.use_diversity,
                "translation_invariant": self.use_ti,
            },
        )


class MIFGSMAttack(TransferIterativeAttack):
    attack_name = "mi-fgsm"
    use_momentum = True


class NIFGSMAttack(TransferIterativeAttack):
    attack_name = "ni-fgsm"
    use_momentum = True
    use_nesterov = True


class DI2FGSMAttack(TransferIterativeAttack):
    attack_name = "di2-fgsm"
    use_diversity = True


class TIFGSMAttack(TransferIterativeAttack):
    attack_name = "ti-fgsm"
    use_ti = True


class NIDITIFGSMAttack(TransferIterativeAttack):
    attack_name = "ni+di+ti"
    use_momentum = True
    use_nesterov = True
    use_diversity = True
    use_ti = True


class TASSAttack(NIDITIFGSMAttack):
    """TASS repository configuration built around NI + DI + TI for segmentation."""

    attack_name = "tass"
