from __future__ import annotations

from itertools import combinations

import torch
import torch.nn.functional as F

from src.attacks.base import AttackOutput, SegmentationAttack
from src.attacks.constraints import project_linf
from src.attacks.losses import build_safe_targets, build_valid_mask, per_image_segmentation_accuracy
from src.attacks.utils import select_feature_map


class RPPGDAttack(SegmentationAttack):
    """Region-and-Prototype PGD adapted from the public RP-PGD implementation."""

    attack_name = "rppgd"

    STATE_IGNORE = 0
    STATE_TRUE = 1
    STATE_FALSE = 2
    STATE_BOUNDARY = 3

    def feature_key(self) -> str | None:
        raw = self.config.extra.get("feature_key")
        return None if raw in (None, "") else str(raw)

    def prototype_similarity_threshold(self) -> float:
        return float(self.config.extra.get("prototype_similarity_threshold", 0.9))

    def region_weight(self) -> float:
        return float(self.config.extra.get("region_weight", 0.5))

    def inter_weight(self) -> float:
        return float(self.config.extra.get("inter_weight", 0.25))

    def intra_weight(self) -> float:
        return float(self.config.extra.get("intra_weight", 0.0))

    def background_ids(self) -> tuple[int, ...]:
        raw = self.config.extra.get("background_ids", (0,))
        if isinstance(raw, (list, tuple, set)):
            return tuple(int(value) for value in raw)
        return (int(raw),)

    @staticmethod
    def _masked_mean(loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = loss_map.masked_select(mask)
        if selected.numel() == 0:
            return loss_map.sum() * 0.0
        return selected.mean()

    def _masked_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = build_valid_mask(
            targets=targets,
            num_classes=logits.size(1),
            ignore_index=self.config.ignore_index,
        )
        safe_targets = build_safe_targets(targets=targets, valid_mask=valid_mask)
        loss_map = F.cross_entropy(logits, safe_targets, reduction="none")
        return self._masked_mean(loss_map, mask & valid_mask)

    @staticmethod
    def _cosine_similarity_score(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (F.cosine_similarity(left, right, dim=0) + 1.0) / 2.0

    def _prepare_feature_map(
        self,
        features: dict[str, torch.Tensor],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        feature_map = select_feature_map(features, preferred_key=self.feature_key())
        if feature_map.ndim != 4:
            raise ValueError(f"RP-PGD expects a 4D feature map, got {tuple(feature_map.shape)}.")
        if tuple(feature_map.shape[-2:]) != tuple(target_size):
            feature_map = F.interpolate(feature_map, size=target_size, mode="bilinear", align_corners=False)
        return feature_map

    def _compute_region_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        state: torch.Tensor,
        step_index: int,
        total_steps: int,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, int]]:
        step_number = step_index + 1
        total_steps = max(int(total_steps), 1)
        lambda_true = 1.0 - float(step_number) / float(2 * total_steps)
        lambda_boundary = float(step_number - 1) / float(2 * total_steps)
        lambda_false = 1.0 / float(2 * total_steps)

        true_mask = state == self.STATE_TRUE
        false_mask = state == self.STATE_FALSE
        boundary_mask = state == self.STATE_BOUNDARY

        loss_true = self._masked_cross_entropy(logits, targets, true_mask)
        loss_false = self._masked_cross_entropy(logits, targets, false_mask)
        loss_boundary = self._masked_cross_entropy(logits, targets, boundary_mask)
        region_loss = lambda_true * loss_true + lambda_boundary * loss_boundary + lambda_false * loss_false

        return region_loss, {
            "true_loss": float(loss_true.detach().cpu().item()),
            "false_loss": float(loss_false.detach().cpu().item()),
            "boundary_loss": float(loss_boundary.detach().cpu().item()),
            "lambda_true": lambda_true,
            "lambda_boundary": lambda_boundary,
            "lambda_false": lambda_false,
        }, {
            "true_pixels": int(true_mask.sum().detach().cpu().item()),
            "false_pixels": int(false_mask.sum().detach().cpu().item()),
            "boundary_pixels": int(boundary_mask.sum().detach().cpu().item()),
        }

    def _compute_prototype_losses(
        self,
        feature_map: torch.Tensor,
        predicted_labels: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
        selected_labels = predicted_labels.detach().clone()
        selected_labels[state != self.STATE_TRUE] = -1
        feature_pixels = feature_map.permute(0, 2, 3, 1)

        prototypes: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for class_id_tensor in torch.unique(selected_labels):
            class_id = int(class_id_tensor.item())
            if class_id < 0:
                continue
            mask = selected_labels == class_id
            if not mask.any():
                continue
            class_features = feature_pixels[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append((class_id, class_features, prototype))

        if len(prototypes) > 1:
            pair_scores = [
                self._cosine_similarity_score(left[2], right[2])
                for left, right in combinations(prototypes, 2)
            ]
            inter_loss = torch.stack(pair_scores).mean()
        else:
            inter_loss = feature_map.sum() * 0.0

        intra_terms: list[torch.Tensor] = []
        background_ids = set(self.background_ids())
        threshold = self.prototype_similarity_threshold()
        for class_id, class_features, prototype in prototypes:
            if class_id in background_ids:
                continue
            cosine_scores = (F.cosine_similarity(class_features, prototype.unsqueeze(0), dim=1) + 1.0) / 2.0
            high_mask = cosine_scores >= threshold
            low_mask = ~high_mask
            loss_high = cosine_scores[high_mask].mean() if high_mask.any() else cosine_scores.sum() * 0.0
            loss_low = cosine_scores[low_mask].mean() if low_mask.any() else cosine_scores.sum() * 0.0
            intra_terms.append(loss_low * 0.75 + loss_high * 0.25)

        if intra_terms:
            intra_loss = torch.stack(intra_terms).mean()
        else:
            intra_loss = feature_map.sum() * 0.0

        return inter_loss, intra_loss, {
            "num_prototypes": len(prototypes),
            "inter_loss": float(inter_loss.detach().cpu().item()),
            "intra_loss": float(intra_loss.detach().cpu().item()),
        }

    def run(self, images: torch.Tensor, targets: torch.Tensor) -> AttackOutput:
        clean = images.detach().clone().to(self.model.device)
        labels = targets.detach().clone().to(self.model.device)
        epsilon = self.config.epsilon
        step_size = self.config.resolved_step_size()
        steps = self.config.steps
        direction = -1.0 if self.config.targeted else 1.0

        valid_mask = build_valid_mask(
            targets=labels,
            num_classes=self.model.num_classes,
            ignore_index=self.config.ignore_index,
        )
        state = torch.full_like(labels, fill_value=self.STATE_IGNORE, dtype=torch.long)
        state[valid_mask] = self.STATE_TRUE

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
            initial_logits = self.model.logits(adversarial)
            initial_prediction = initial_logits.argmax(dim=1)
            state[(initial_prediction != labels) & valid_mask] = self.STATE_FALSE

        last_region_stats: dict[str, float] = {}
        last_region_counts: dict[str, int] = {}
        last_proto_stats: dict[str, float | int] = {}
        last_total_loss = 0.0

        with torch.enable_grad():
            for step_index in range(steps):
                adversarial = adversarial.detach().requires_grad_(True)
                logits, features = self.model.forward_with_features(adversarial)
                prediction = logits.argmax(dim=1)
                state[(prediction != labels) & (state == self.STATE_TRUE) & valid_mask] = self.STATE_BOUNDARY

                region_loss, last_region_stats, last_region_counts = self._compute_region_loss(
                    logits=logits,
                    targets=labels,
                    state=state,
                    step_index=step_index,
                    total_steps=steps,
                )
                feature_map = self._prepare_feature_map(features, target_size=tuple(labels.shape[-2:]))
                inter_loss, intra_loss, last_proto_stats = self._compute_prototype_losses(
                    feature_map=feature_map,
                    predicted_labels=prediction,
                    state=state,
                )
                total_loss = (
                    self.region_weight() * region_loss
                    + self.inter_weight() * inter_loss
                    + self.intra_weight() * intra_loss
                )
                objective = direction * total_loss
                gradient = torch.autograd.grad(objective, adversarial, retain_graph=False, create_graph=False)[0]
                adversarial = adversarial.detach() + step_size * gradient.sign()
                adversarial = project_linf(
                    adversarial_images=adversarial,
                    clean_images=clean,
                    epsilon=epsilon,
                    min_value=self.config.clamp_min,
                    max_value=self.config.clamp_max,
                )
                last_total_loss = float(total_loss.detach().cpu().item())

        with torch.no_grad():
            final_logits = self.model.logits(adversarial)
            final_accuracy = per_image_segmentation_accuracy(
                logits=final_logits,
                targets=labels,
                ignore_index=self.config.ignore_index,
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
                "loss": last_total_loss,
                "random_start": self.config.random_start,
                "targeted": self.config.targeted,
                "feature_key": self.feature_key() or "",
                "prototype_similarity_threshold": self.prototype_similarity_threshold(),
                "region_weight": self.region_weight(),
                "inter_weight": self.inter_weight(),
                "intra_weight": self.intra_weight(),
                "region": {
                    **last_region_stats,
                    **last_region_counts,
                },
                "prototype": last_proto_stats,
                "best_mean_score": float(final_accuracy.mean().cpu().item()),
            },
        )
