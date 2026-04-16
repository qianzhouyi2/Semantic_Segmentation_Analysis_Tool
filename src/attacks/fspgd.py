from __future__ import annotations

import torch
import torch.nn.functional as F

from src.attacks.base import AttackOutput, SegmentationAttack
from src.attacks.constraints import project_linf
from src.attacks.utils import cosine_threshold, select_feature_map


class FSPGDAttack(SegmentationAttack):
    """Feature Similarity PGD using intermediate feature affinities."""

    attack_name = "fspgd"

    def cosine_bins(self) -> float:
        return float(self.config.extra.get("cosine_bins", 3.0))

    def feature_key(self) -> str | None:
        raw = self.config.extra.get("feature_key")
        return None if raw in (None, "") else str(raw)

    def _prepare_feature_affinity(
        self,
        feature_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_feature = F.normalize(feature_map.detach(), dim=1)
        flat_feature = normalized_feature.flatten(2)
        affinity = torch.bmm(flat_feature.transpose(1, 2), flat_feature)
        num_positions = affinity.size(-1)
        off_diagonal = 1.0 - torch.eye(num_positions, device=feature_map.device, dtype=affinity.dtype).unsqueeze(0)
        threshold = cosine_threshold(self.cosine_bins())
        weight = ((affinity * off_diagonal) > threshold).to(dtype=feature_map.dtype)
        return flat_feature, weight

    def _objective(
        self,
        attack_input: torch.Tensor,
        *,
        clean_feature_map: torch.Tensor,
        clean_feature_flat: torch.Tensor,
        affinity_weight: torch.Tensor,
        pair_count: torch.Tensor,
        step_index: int,
        steps: int,
        direction: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        _, adv_features = self.model.forward_with_features(attack_input)
        adv_feature_map = select_feature_map(adv_features, preferred_key=self.feature_key())
        if adv_feature_map.shape[-2:] != clean_feature_map.shape[-2:]:
            adv_feature_map = F.interpolate(
                adv_feature_map,
                size=clean_feature_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        adv_feature_flat = adv_feature_map.flatten(2)
        diagonal_similarity = F.cosine_similarity(clean_feature_flat, adv_feature_flat, dim=1).mean(dim=1)
        combinational_similarity = (
            affinity_weight * torch.bmm(adv_feature_flat.transpose(1, 2), adv_feature_flat)
        ).sum(dim=(1, 2)) / pair_count / 2.0
        lambda_t = float(step_index + 1) / float(max(steps, 1))
        loss_per_image = -lambda_t * diagonal_similarity - (1.0 - lambda_t) * combinational_similarity
        loss = loss_per_image.mean()
        return direction * loss, {"loss": float(loss.detach().cpu().item())}

    def run(self, images: torch.Tensor, targets: torch.Tensor) -> AttackOutput:
        del targets
        epsilon = self.config.epsilon
        step_size = self.config.resolved_step_size()
        steps = self.config.steps
        direction = -1.0 if self.config.targeted else 1.0

        clean = images.detach().clone().to(self.model.device)
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
            _, clean_features = self.model.forward_with_features(clean)
            clean_feature_map = select_feature_map(clean_features, preferred_key=self.feature_key())
        clean_feature_flat, affinity_weight = self._prepare_feature_affinity(clean_feature_map)
        pair_count = affinity_weight.sum(dim=(1, 2)).clamp_min(1.0)

        loss_value = 0.0
        with torch.enable_grad():
            for step_index in range(steps):
                gradient, stats = self.estimate_input_gradient(
                    adversarial,
                    lambda attack_input: self._objective(
                        attack_input=attack_input,
                        clean_feature_map=clean_feature_map,
                        clean_feature_flat=clean_feature_flat,
                        affinity_weight=affinity_weight,
                        pair_count=pair_count,
                        step_index=step_index,
                        steps=steps,
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
                "epsilon": epsilon,
                "loss": loss_value,
                "feature_key": self.feature_key() or "",
                "cosine_bins": self.cosine_bins(),
                "random_start": self.config.random_start,
                "targeted": self.config.targeted,
            },
        )
