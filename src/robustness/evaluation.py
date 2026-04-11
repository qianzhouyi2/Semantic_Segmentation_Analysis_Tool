from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RobustnessSummary:
    clean_pixel_accuracy: float
    adv_pixel_accuracy: float
    clean_mean_iou: float
    adv_mean_iou: float
    clean_mean_dice: float
    adv_mean_dice: float

    @property
    def pixel_accuracy_drop(self) -> float:
        return self.clean_pixel_accuracy - self.adv_pixel_accuracy

    @property
    def mean_iou_drop(self) -> float:
        return self.clean_mean_iou - self.adv_mean_iou

    @property
    def mean_dice_drop(self) -> float:
        return self.clean_mean_dice - self.adv_mean_dice

    def to_dict(self) -> dict[str, float]:
        return {
            "clean_pixel_accuracy": self.clean_pixel_accuracy,
            "adv_pixel_accuracy": self.adv_pixel_accuracy,
            "pixel_accuracy_drop": self.pixel_accuracy_drop,
            "clean_mean_iou": self.clean_mean_iou,
            "adv_mean_iou": self.adv_mean_iou,
            "mean_iou_drop": self.mean_iou_drop,
            "clean_mean_dice": self.clean_mean_dice,
            "adv_mean_dice": self.adv_mean_dice,
            "mean_dice_drop": self.mean_dice_drop,
        }


def compare_clean_and_adversarial(clean_summary: dict[str, Any], adv_summary: dict[str, Any]) -> RobustnessSummary:
    clean_metrics = clean_summary["metrics"]
    adv_metrics = adv_summary["metrics"]
    return RobustnessSummary(
        clean_pixel_accuracy=float(clean_metrics["pixel_accuracy"]),
        adv_pixel_accuracy=float(adv_metrics["pixel_accuracy"]),
        clean_mean_iou=float(clean_metrics["mean_iou"]),
        adv_mean_iou=float(adv_metrics["mean_iou"]),
        clean_mean_dice=float(clean_metrics["mean_dice"]),
        adv_mean_dice=float(adv_metrics["mean_dice"]),
    )
