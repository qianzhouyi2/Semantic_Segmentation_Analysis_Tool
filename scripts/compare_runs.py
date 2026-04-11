from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.comparison.report_diff import compare_metric_reports, load_summary
from src.reporting.exporter import write_csv, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two segmentation evaluation summary files.")
    parser.add_argument("--baseline", required=True, help="Baseline summary.json")
    parser.add_argument("--candidate", required=True, help="Candidate summary.json")
    parser.add_argument("--output-dir", default="results/reports/comparison", help="Directory for outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_summary(args.baseline)
    candidate = load_summary(args.candidate)
    comparison = compare_metric_reports(baseline, candidate)

    output_dir = Path(args.output_dir)
    write_json(output_dir / "comparison.json", comparison)
    write_csv(output_dir / "per_class_deltas.csv", comparison["per_class"])
    write_markdown(
        output_dir / "report.md",
        "Evaluation Comparison",
        [
            f"- pixel_accuracy_delta: {comparison['summary']['pixel_accuracy_delta']:.4f}",
            f"- mean_iou_delta: {comparison['summary']['mean_iou_delta']:.4f}",
            f"- mean_dice_delta: {comparison['summary']['mean_dice_delta']:.4f}",
            f"- mean_precision_delta: {comparison['summary']['mean_precision_delta']:.4f}",
            f"- mean_recall_delta: {comparison['summary']['mean_recall_delta']:.4f}",
            f"- mean_f1_delta: {comparison['summary']['mean_f1_delta']:.4f}",
        ],
    )
    print(f"Comparison report written to: {output_dir}")


if __name__ == "__main__":
    main()
