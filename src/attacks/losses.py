from __future__ import annotations

import torch
import torch.nn.functional as F


def build_valid_mask(
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    valid_mask = (targets >= 0) & (targets < num_classes)
    if ignore_index is not None:
        valid_mask &= targets != ignore_index
    return valid_mask


def build_safe_targets(
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.where(valid_mask, targets.long(), torch.zeros_like(targets, dtype=torch.long))


def spatial_mean(loss_map: torch.Tensor) -> torch.Tensor:
    return loss_map.view(loss_map.size(0), -1).mean(dim=1)


def masked_mean(loss_map: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid_losses = loss_map.masked_select(valid_mask)
    if valid_losses.numel() == 0:
        return loss_map.sum() * 0.0
    return valid_losses.mean()


def build_balanced_class_weights(
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    valid_mask = build_valid_mask(targets=targets, num_classes=num_classes, ignore_index=ignore_index)
    valid_targets = targets.masked_select(valid_mask).long()
    if valid_targets.numel() == 0:
        return torch.ones(num_classes, device=targets.device, dtype=torch.float32)

    counts = torch.bincount(valid_targets, minlength=num_classes).to(device=targets.device, dtype=torch.float32)
    present = counts > 0
    safe_counts = counts.clamp_min(1.0)
    weights = valid_targets.numel() / safe_counts
    weights = torch.where(present, weights, torch.zeros_like(weights))
    present_mean = weights.masked_select(present).mean()
    if torch.isfinite(present_mean) and float(present_mean.item()) > 0.0:
        weights = weights / present_mean
    return weights


def segmentation_attack_loss_map(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str = "cross_entropy",
    ignore_index: int | None = None,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if loss_name != "cross_entropy":
        raise NotImplementedError(f"Unsupported attack loss: {loss_name}")

    num_classes = logits.size(1)
    valid_mask = build_valid_mask(targets=targets, num_classes=num_classes, ignore_index=ignore_index)
    safe_targets = build_safe_targets(targets=targets, valid_mask=valid_mask)
    loss_map = F.cross_entropy(
        logits,
        safe_targets,
        reduction="none",
        weight=None if class_weights is None else class_weights.to(device=logits.device, dtype=logits.dtype),
    )
    return loss_map, valid_mask


def segmentation_attack_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str = "cross_entropy",
    ignore_index: int | None = None,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_map, valid_mask = segmentation_attack_loss_map(
        logits=logits,
        targets=targets,
        loss_name=loss_name,
        ignore_index=ignore_index,
        class_weights=class_weights,
    )
    return masked_mean(loss_map=loss_map, valid_mask=valid_mask)


def segmentation_cospgd_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str = "cross_entropy",
    ignore_index: int | None = None,
    targeted: bool = False,
) -> torch.Tensor:
    loss_map, valid_mask = segmentation_attack_loss_map(
        logits=logits,
        targets=targets,
        loss_name=loss_name,
        ignore_index=ignore_index,
    )
    if not valid_mask.any():
        return logits.sum() * 0.0

    num_classes = logits.size(1)
    safe_targets = build_safe_targets(targets=targets, valid_mask=valid_mask)
    one_hot_targets = F.one_hot(safe_targets.clamp(min=0, max=num_classes - 1), num_classes=num_classes)
    one_hot_targets = one_hot_targets.permute(0, 3, 1, 2).to(dtype=logits.dtype)
    cosine_similarity = F.cosine_similarity(F.softmax(logits, dim=1), one_hot_targets, dim=1)
    if targeted:
        cosine_similarity = 1.0 - cosine_similarity
    scaled_loss_map = cosine_similarity.detach() * loss_map
    return masked_mean(loss_map=scaled_loss_map, valid_mask=valid_mask)


def segmentation_segpgd_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    iteration: int,
    iterations: int,
    loss_name: str = "cross_entropy",
    ignore_index: int | None = None,
    targeted: bool = False,
) -> torch.Tensor:
    loss_map, valid_mask = segmentation_attack_loss_map(
        logits=logits,
        targets=targets,
        loss_name=loss_name,
        ignore_index=ignore_index,
    )
    if not valid_mask.any():
        return logits.sum() * 0.0

    safe_targets = build_safe_targets(targets=targets, valid_mask=valid_mask)
    pred = logits.argmax(dim=1)
    lambda_t = float(iteration) / float(max(2 * iterations, 1))
    if targeted:
        weights = torch.where(pred == safe_targets, lambda_t, 1.0 - lambda_t)
    else:
        weights = torch.where(pred == safe_targets, 1.0 - lambda_t, lambda_t)
    scaled_loss_map = weights.to(dtype=loss_map.dtype) * loss_map
    return masked_mean(loss_map=scaled_loss_map, valid_mask=valid_mask)


def segmentation_masked_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int | None = None,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_map, valid_mask = segmentation_attack_loss_map(
        logits=logits,
        targets=targets,
        ignore_index=ignore_index,
        class_weights=class_weights,
    )
    safe_targets = build_safe_targets(targets=targets, valid_mask=valid_mask)
    correct_mask = valid_mask & (logits.argmax(dim=1) == safe_targets)
    masked_loss_map = correct_mask.float().detach() * loss_map
    return spatial_mean(masked_loss_map).mean()


def segmentation_masked_cross_entropy_balanced_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int | None = None,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = class_weights
    if weights is None:
        weights = build_balanced_class_weights(
            targets=targets,
            num_classes=logits.size(1),
            ignore_index=ignore_index,
        )
    return segmentation_masked_cross_entropy_loss(
        logits=logits,
        targets=targets,
        ignore_index=ignore_index,
        class_weights=weights,
    )


def segmentation_js_divergence_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int | None = None,
) -> torch.Tensor:
    num_classes = logits.size(1)
    valid_mask = build_valid_mask(targets=targets, num_classes=num_classes, ignore_index=ignore_index)
    if not valid_mask.any():
        return logits.sum() * 0.0

    safe_targets = build_safe_targets(targets=targets, valid_mask=valid_mask)
    probs = F.softmax(logits, dim=1)
    target_one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).to(dtype=logits.dtype)
    midpoint = (probs + target_one_hot) / 2.0
    log_midpoint = midpoint.clamp_min(1e-12).log()
    js_map = 0.5 * (
        F.kl_div(log_midpoint, probs, reduction="none")
        + F.kl_div(log_midpoint, target_one_hot, reduction="none")
    ).sum(dim=1)
    js_map = valid_mask.float() * js_map
    return spatial_mean(js_map).mean()


def segmentation_kl_divergence_map(
    adv_logits: torch.Tensor,
    clean_logits: torch.Tensor,
    targets: torch.Tensor | None = None,
    ignore_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if adv_logits.shape != clean_logits.shape:
        raise ValueError("`adv_logits` and `clean_logits` must have the same shape.")

    adv_probs = F.softmax(adv_logits, dim=1)
    clean_probs = F.softmax(clean_logits, dim=1)
    kl_map = (
        adv_probs * (adv_probs.clamp_min(1e-12).log() - clean_probs.clamp_min(1e-12).log())
    ).sum(dim=1)

    if targets is None:
        valid_mask = torch.ones_like(kl_map, dtype=torch.bool)
    else:
        valid_mask = build_valid_mask(
            targets=targets,
            num_classes=adv_logits.size(1),
            ignore_index=ignore_index,
        )
    return kl_map, valid_mask


def per_image_segmentation_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int | None = None,
) -> torch.Tensor:
    valid_mask = build_valid_mask(targets=targets, num_classes=logits.size(1), ignore_index=ignore_index)
    safe_targets = build_safe_targets(targets=targets, valid_mask=valid_mask)
    correct = (logits.argmax(dim=1) == safe_targets) & valid_mask
    numerator = correct.view(correct.size(0), -1).float().sum(dim=1)
    denominator = valid_mask.view(valid_mask.size(0), -1).float().sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def sample_alternate_labels(
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    valid_mask = build_valid_mask(targets=targets, num_classes=num_classes, ignore_index=ignore_index)
    safe_targets = build_safe_targets(targets=targets, valid_mask=valid_mask)
    random_offset = torch.randint(1, num_classes, size=targets.shape, device=targets.device)
    alternate = (safe_targets + random_offset) % num_classes
    fallback_value = ignore_index if ignore_index is not None else 0
    return torch.where(valid_mask, alternate, torch.full_like(alternate, fallback_value))
