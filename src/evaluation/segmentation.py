from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.metrics.segmentation import compute_confusion_matrix, summarize_confusion_matrix, summarize_per_sample_confusion_matrix


@torch.no_grad()
def evaluate_segmentation_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: str | torch.device = "cuda",
    ignore_index: int | None = None,
    class_names: dict[int, str] | None = None,
    max_batches: int = -1,
    logger: Any | None = None,
    log_interval: int = 20,
    collect_per_sample: bool = False,
) -> dict[str, Any]:
    model.eval()
    model.to(device)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    processed_batches = 0
    processed_samples = 0
    filenames: list[str] = []
    per_sample_metrics: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(dataloader):
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError(f"Unexpected batch format: {type(batch)!r}")

        images = batch[0].to(device, non_blocking=True)
        targets = batch[1].cpu().numpy()
        batch_filenames: list[str] = []
        if len(batch) >= 3 and isinstance(batch[2], Iterable):
            batch_filenames = [str(item) for item in batch[2]]
            filenames.extend(batch_filenames)
        elif collect_per_sample:
            batch_filenames = [f"sample_{processed_samples + offset:06d}" for offset in range(images.shape[0])]

        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().numpy()

        for sample_offset, (target, prediction) in enumerate(zip(targets, predictions, strict=True)):
            sample_confusion = compute_confusion_matrix(
                target=target,
                prediction=prediction,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
            confusion += sample_confusion
            if collect_per_sample:
                sample_metrics = summarize_per_sample_confusion_matrix(sample_confusion)
                filename = (
                    batch_filenames[sample_offset]
                    if sample_offset < len(batch_filenames)
                    else f"sample_{processed_samples + sample_offset:06d}"
                )
                per_sample_metrics.append(
                    {
                        "sample_index": processed_samples + sample_offset,
                        "filename": filename,
                        "pixel_accuracy": sample_metrics.pixel_accuracy,
                        "sample_miou": sample_metrics.sample_miou,
                        "sample_dice": sample_metrics.sample_dice,
                        "valid_class_count": sample_metrics.valid_class_count,
                    }
                )

        processed_batches += 1
        processed_samples += images.shape[0]
        if logger is not None and (processed_batches == 1 or processed_batches % log_interval == 0):
            logger.info("Evaluation progress: batches=%d samples=%d", processed_batches, processed_samples)
        if max_batches > 0 and batch_index + 1 >= max_batches:
            break

    metrics = summarize_confusion_matrix(confusion, class_names=class_names)
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
    }
    if collect_per_sample:
        payload["per_sample_metrics"] = per_sample_metrics
    return payload
