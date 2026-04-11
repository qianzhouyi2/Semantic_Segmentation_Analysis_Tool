from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_summary(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_metric_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]

    summary = {
        "pixel_accuracy_delta": candidate_metrics["pixel_accuracy"] - baseline_metrics["pixel_accuracy"],
        "mean_iou_delta": candidate_metrics["mean_iou"] - baseline_metrics["mean_iou"],
        "mean_dice_delta": candidate_metrics["mean_dice"] - baseline_metrics["mean_dice"],
        "mean_precision_delta": candidate_metrics["mean_precision"] - baseline_metrics["mean_precision"],
        "mean_recall_delta": candidate_metrics["mean_recall"] - baseline_metrics["mean_recall"],
        "mean_f1_delta": candidate_metrics["mean_f1"] - baseline_metrics["mean_f1"],
    }

    baseline_rows = {int(row["class_id"]): row for row in baseline_metrics["per_class"]}
    candidate_rows = {int(row["class_id"]): row for row in candidate_metrics["per_class"]}
    per_class = []
    for class_id in sorted(set(baseline_rows) | set(candidate_rows)):
        base_row = baseline_rows.get(class_id, {})
        cand_row = candidate_rows.get(class_id, {})
        per_class.append(
            {
                "class_id": class_id,
                "class_name": cand_row.get("class_name", base_row.get("class_name", f"class_{class_id}")),
                "iou_delta": float(cand_row.get("iou", 0.0)) - float(base_row.get("iou", 0.0)),
                "dice_delta": float(cand_row.get("dice", 0.0)) - float(base_row.get("dice", 0.0)),
                "precision_delta": float(cand_row.get("precision", 0.0)) - float(base_row.get("precision", 0.0)),
                "recall_delta": float(cand_row.get("recall", 0.0)) - float(base_row.get("recall", 0.0)),
                "f1_delta": float(cand_row.get("f1", 0.0)) - float(base_row.get("f1", 0.0)),
            }
        )

    return {
        "summary": summary,
        "per_class": per_class,
    }
