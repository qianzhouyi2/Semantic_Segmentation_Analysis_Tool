from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import _bootstrap  # noqa: F401

from src.common.config import load_label_config
from src.datasets.scanner import discover_files
from src.io.image_io import DEFAULT_MASK_SUFFIXES, load_mask
from src.metrics.segmentation import compute_confusion_matrix, summarize_confusion_matrix
from src.reporting.exporter import write_csv, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions against ground truth masks.")
    parser.add_argument("--gt-dir", required=True, help="Ground truth mask directory.")
    parser.add_argument("--pred-dir", required=True, help="Prediction mask directory.")
    parser.add_argument("--label-config", default="", help="Optional YAML file with class IDs and names.")
    parser.add_argument("--num-classes", type=int, default=0, help="Class count when no label config is provided.")
    parser.add_argument("--ignore-index", type=int, default=-1, help="Ignore label. Use -1 to disable.")
    parser.add_argument("--output-dir", default="results/reports/evaluation", help="Directory for outputs.")
    return parser.parse_args()


def infer_num_classes(paths: list[Path], ignore_index: int | None) -> int:
    max_label = -1
    for path in paths:
        mask = load_mask(path)
        if ignore_index is not None:
            mask = mask[mask != ignore_index]
        if mask.size == 0:
            continue
        max_label = max(max_label, int(mask.max()))
    if max_label < 0:
        raise ValueError("Unable to infer class count from empty masks.")
    return max_label + 1


def main() -> None:
    args = parse_args()

    label_config = load_label_config(args.label_config) if args.label_config else None
    class_names = label_config.class_names if label_config else {}
    ignore_index = label_config.ignore_index if label_config else args.ignore_index
    if ignore_index == -1:
        ignore_index = None

    gt_files = discover_files(args.gt_dir, DEFAULT_MASK_SUFFIXES)
    pred_files = discover_files(args.pred_dir, DEFAULT_MASK_SUFFIXES)
    matched_keys = sorted(set(gt_files) & set(pred_files))
    missing_predictions = sorted(set(gt_files) - set(pred_files))
    orphan_predictions = sorted(set(pred_files) - set(gt_files))

    if not matched_keys:
        raise ValueError("No matched prediction / ground-truth pairs were found.")

    num_classes = len(label_config.class_ids) if label_config else args.num_classes
    if num_classes <= 0:
        num_classes = infer_num_classes(
            [gt_files[key] for key in matched_keys] + [pred_files[key] for key in matched_keys],
            ignore_index=ignore_index,
        )

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    shape_mismatches: list[str] = []
    for key in matched_keys:
        gt_mask = load_mask(gt_files[key])
        pred_mask = load_mask(pred_files[key])
        if gt_mask.shape != pred_mask.shape:
            shape_mismatches.append(key)
            continue
        confusion += compute_confusion_matrix(gt_mask, pred_mask, num_classes=num_classes, ignore_index=ignore_index)

    metrics = summarize_confusion_matrix(confusion, class_names=class_names)
    output_dir = Path(args.output_dir)
    payload = {
        "matched_pairs": len(matched_keys),
        "missing_predictions": missing_predictions,
        "orphan_predictions": orphan_predictions,
        "shape_mismatches": shape_mismatches,
        "metrics": metrics.to_dict(),
    }

    write_json(output_dir / "summary.json", payload)
    write_csv(output_dir / "per_class_metrics.csv", metrics.per_class)
    write_markdown(
        output_dir / "report.md",
        "Segmentation Evaluation",
        [
            f"- matched_pairs: {len(matched_keys)}",
            f"- missing_predictions: {len(missing_predictions)}",
            f"- orphan_predictions: {len(orphan_predictions)}",
            f"- shape_mismatches: {len(shape_mismatches)}",
            "",
            "## Overall",
            f"- pixel_accuracy: {metrics.pixel_accuracy:.4f}",
            f"- mean_iou: {metrics.mean_iou:.4f}",
            f"- mean_dice: {metrics.mean_dice:.4f}",
            f"- mean_precision: {metrics.mean_precision:.4f}",
            f"- mean_recall: {metrics.mean_recall:.4f}",
            f"- mean_f1: {metrics.mean_f1:.4f}",
        ],
    )

    print(f"Matched pairs: {len(matched_keys)}")
    print(f"mIoU: {metrics.mean_iou:.4f}")
    print(f"Pixel Accuracy: {metrics.pixel_accuracy:.4f}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
