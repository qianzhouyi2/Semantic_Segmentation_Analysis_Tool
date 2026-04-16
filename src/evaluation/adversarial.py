from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.attacks import (
    AttackConfig,
    AttackRunner,
    finalize_attack_runtime_aggregate,
    init_attack_runtime_aggregate,
    update_attack_runtime_aggregate,
)
from src.metrics.segmentation import compute_confusion_matrix, summarize_confusion_matrix
from src.models.base import SegmentationModelAdapter


def evaluate_adversarial_segmentation_model(
    model: SegmentationModelAdapter,
    attack_config: AttackConfig,
    dataloader: DataLoader,
    ignore_index: int | None = None,
    class_names: dict[int, str] | None = None,
    max_batches: int = -1,
    logger: Any | None = None,
    log_interval: int = 20,
) -> dict[str, Any]:
    confusion = np.zeros((model.num_classes, model.num_classes), dtype=np.int64)
    attack_runner = AttackRunner(model)

    processed_batches = 0
    processed_samples = 0
    filenames: list[str] = []
    linf_values: list[torch.Tensor] = []
    l2_values: list[torch.Tensor] = []
    attack_runtime_aggregate = init_attack_runtime_aggregate(attack_config)

    for batch_index, batch in enumerate(dataloader):
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError(f"Unexpected batch format: {type(batch)!r}")

        images = batch[0].to(model.device, non_blocking=True)
        targets = batch[1].to(model.device, non_blocking=True)
        targets_cpu = targets.cpu().numpy()
        if len(batch) >= 3 and isinstance(batch[2], Iterable):
            filenames.extend([str(item) for item in batch[2]])

        attack_output = attack_runner.run(config=attack_config, images=images, targets=targets)
        update_attack_runtime_aggregate(
            attack_runtime_aggregate,
            dict(attack_output.metadata),
            batch_size=images.shape[0],
        )
        with torch.no_grad():
            predictions = model.predict(attack_output.adversarial_images).cpu().numpy()

        perturbation = attack_output.perturbation.detach()
        linf_values.append(perturbation.abs().flatten(1).amax(dim=1).cpu())
        l2_values.append(perturbation.flatten(1).norm(p=2, dim=1).cpu())

        for target, prediction in zip(targets_cpu, predictions, strict=True):
            confusion += compute_confusion_matrix(
                target=target,
                prediction=prediction,
                num_classes=model.num_classes,
                ignore_index=ignore_index,
            )

        processed_batches += 1
        processed_samples += images.shape[0]
        if logger is not None and (processed_batches == 1 or processed_batches % log_interval == 0):
            logger.info("Adversarial evaluation progress: batches=%d samples=%d", processed_batches, processed_samples)
        if max_batches > 0 and batch_index + 1 >= max_batches:
            break

    metrics = summarize_confusion_matrix(confusion, class_names=class_names)
    linf_tensor = torch.cat(linf_values) if linf_values else torch.empty(0)
    l2_tensor = torch.cat(l2_values) if l2_values else torch.empty(0)
    attack_runtime_metadata = finalize_attack_runtime_aggregate(attack_runtime_aggregate)
    payload = {
        "processed_batches": processed_batches,
        "processed_samples": processed_samples,
        "filenames": filenames,
        "metrics": metrics.to_dict(),
        "reference": {
            "mAcc": float(metrics.mean_recall),
            "aAcc": float(metrics.pixel_accuracy),
            "mIoU": float(metrics.mean_iou),
        },
        "reference_percent": {
            "mAcc": float(metrics.mean_recall * 100.0),
            "aAcc": float(metrics.pixel_accuracy * 100.0),
            "mIoU": float(metrics.mean_iou * 100.0),
        },
        "attack": {
            "name": attack_config.name,
            "epsilon": attack_config.epsilon,
            "step_size": attack_config.resolved_step_size(),
            "steps": attack_config.steps,
            "random_start": attack_config.random_start,
            "targeted": attack_config.targeted,
            "loss_name": attack_config.loss_name,
            "ignore_index": attack_config.ignore_index,
            "mean_linf": float(linf_tensor.mean().item()) if linf_tensor.numel() else 0.0,
            "max_linf": float(linf_tensor.max().item()) if linf_tensor.numel() else 0.0,
            "mean_l2": float(l2_tensor.mean().item()) if l2_tensor.numel() else 0.0,
            **attack_config.protocol_metadata(),
            **attack_runtime_metadata,
            "extra": attack_config.extra,
        },
    }
    return payload
