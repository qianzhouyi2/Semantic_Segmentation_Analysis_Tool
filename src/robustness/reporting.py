from __future__ import annotations

from src.robustness.evaluation import RobustnessSummary


def build_robustness_report_lines(summary: RobustnessSummary) -> list[str]:
    return [
        "## Clean vs Adversarial",
        f"- clean_pixel_accuracy: {summary.clean_pixel_accuracy:.4f}",
        f"- adv_pixel_accuracy: {summary.adv_pixel_accuracy:.4f}",
        f"- pixel_accuracy_drop: {summary.pixel_accuracy_drop:.4f}",
        f"- clean_mean_iou: {summary.clean_mean_iou:.4f}",
        f"- adv_mean_iou: {summary.adv_mean_iou:.4f}",
        f"- mean_iou_drop: {summary.mean_iou_drop:.4f}",
        f"- clean_mean_dice: {summary.clean_mean_dice:.4f}",
        f"- adv_mean_dice: {summary.adv_mean_dice:.4f}",
        f"- mean_dice_drop: {summary.mean_dice_drop:.4f}",
    ]
