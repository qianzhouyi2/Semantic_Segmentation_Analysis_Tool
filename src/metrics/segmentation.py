from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _safe_divide(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    result = np.full_like(numerator_array, np.nan, dtype=np.float64)
    valid = denominator_array != 0
    result[valid] = numerator_array[valid] / denominator_array[valid]
    return result


def compute_confusion_matrix(
    target: np.ndarray,
    prediction: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> np.ndarray:
    if target.shape != prediction.shape:
        raise ValueError("Target and prediction masks must have identical shapes.")

    valid = (target >= 0) & (target < num_classes) & (prediction >= 0) & (prediction < num_classes)
    if ignore_index is not None:
        valid &= target != ignore_index

    encoded = num_classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    histogram = np.bincount(encoded, minlength=num_classes * num_classes)
    return histogram.reshape(num_classes, num_classes)


def _compute_class_statistics(
    confusion_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_positive = np.diag(confusion_matrix).astype(np.float64)
    gt_pixels = confusion_matrix.sum(axis=1).astype(np.float64)
    pred_pixels = confusion_matrix.sum(axis=0).astype(np.float64)
    union = gt_pixels + pred_pixels - true_positive

    precision = _safe_divide(true_positive, pred_pixels)
    recall = _safe_divide(true_positive, gt_pixels)
    iou = _safe_divide(true_positive, union)
    dice = _safe_divide(2.0 * true_positive, gt_pixels + pred_pixels)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return true_positive, gt_pixels, pred_pixels, union, precision, recall, iou, dice, f1


@dataclass(slots=True)
class SegmentationMetrics:
    confusion_matrix: np.ndarray
    pixel_accuracy: float
    mean_iou: float
    mean_dice: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    per_class: list[dict[str, float | int | str]]

    def to_dict(self) -> dict[str, object]:
        return {
            "pixel_accuracy": self.pixel_accuracy,
            "mean_iou": self.mean_iou,
            "mean_dice": self.mean_dice,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "mean_f1": self.mean_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


@dataclass(slots=True)
class PerSampleSegmentationMetrics:
    confusion_matrix: np.ndarray
    pixel_accuracy: float
    sample_miou: float
    sample_dice: float
    valid_class_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "pixel_accuracy": self.pixel_accuracy,
            "sample_miou": self.sample_miou,
            "sample_dice": self.sample_dice,
            "valid_class_count": self.valid_class_count,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


def summarize_per_sample_confusion_matrix(confusion_matrix: np.ndarray) -> PerSampleSegmentationMetrics:
    """Summarize a single-image confusion matrix.

    The image-wise mIoU is defined as the mean IoU over classes whose union is
    non-empty within this sample. Classes with zero union are excluded so empty
    classes do not dilute the per-image score.
    """

    (
        true_positive,
        _gt_pixels,
        _pred_pixels,
        union,
        _precision,
        _recall,
        iou,
        dice,
        _f1,
    ) = _compute_class_statistics(confusion_matrix)
    valid_classes = union > 0
    total_pixels = float(confusion_matrix.sum())
    return PerSampleSegmentationMetrics(
        confusion_matrix=confusion_matrix,
        pixel_accuracy=float(true_positive.sum() / total_pixels) if total_pixels else 0.0,
        sample_miou=float(np.nanmean(iou[valid_classes])) if np.any(valid_classes) else 0.0,
        sample_dice=float(np.nanmean(dice[valid_classes])) if np.any(valid_classes) else 0.0,
        valid_class_count=int(valid_classes.sum()),
    )


def compute_per_sample_segmentation_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> PerSampleSegmentationMetrics:
    confusion_matrix = compute_confusion_matrix(
        target=target,
        prediction=prediction,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    return summarize_per_sample_confusion_matrix(confusion_matrix)


def summarize_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: dict[int, str] | None = None,
) -> SegmentationMetrics:
    class_names = class_names or {}
    true_positive, gt_pixels, pred_pixels, _union, precision, recall, iou, dice, f1 = _compute_class_statistics(
        confusion_matrix
    )

    per_class: list[dict[str, float | int | str]] = []
    for class_id in range(confusion_matrix.shape[0]):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, f"class_{class_id}"),
                "gt_pixels": int(gt_pixels[class_id]),
                "pred_pixels": int(pred_pixels[class_id]),
                "iou": float(iou[class_id]) if not np.isnan(iou[class_id]) else float("nan"),
                "dice": float(dice[class_id]) if not np.isnan(dice[class_id]) else float("nan"),
                "precision": float(precision[class_id]) if not np.isnan(precision[class_id]) else float("nan"),
                "recall": float(recall[class_id]) if not np.isnan(recall[class_id]) else float("nan"),
                "f1": float(f1[class_id]) if not np.isnan(f1[class_id]) else float("nan"),
            }
        )

    total_pixels = float(confusion_matrix.sum())
    return SegmentationMetrics(
        confusion_matrix=confusion_matrix,
        pixel_accuracy=float(true_positive.sum() / total_pixels) if total_pixels else 0.0,
        mean_iou=float(np.nanmean(iou)) if iou.size else 0.0,
        mean_dice=float(np.nanmean(dice)) if dice.size else 0.0,
        mean_precision=float(np.nanmean(precision)) if precision.size else 0.0,
        mean_recall=float(np.nanmean(recall)) if recall.size else 0.0,
        mean_f1=float(np.nanmean(f1)) if f1.size else 0.0,
        per_class=per_class,
    )
