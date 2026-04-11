from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.common.config import load_dataset_config, load_label_config
from src.datasets.scanner import scan_dataset
from src.datasets.stats import compute_class_statistics
from src.reporting.exporter import write_csv, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run basic dataset health checks for segmentation data.")
    parser.add_argument("--dataset-config", default="", help="YAML file with image_dir and mask_dir.")
    parser.add_argument("--label-config", default="", help="Optional YAML file with class IDs and colors.")
    parser.add_argument("--image-dir", default="", help="Image directory when dataset config is not used.")
    parser.add_argument("--mask-dir", default="", help="Mask directory when dataset config is not used.")
    parser.add_argument("--output-dir", default="results/reports/dataset_check", help="Directory for outputs.")
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace):
    if args.dataset_config:
        dataset_config = load_dataset_config(args.dataset_config)
        image_dir = dataset_config.image_dir
        mask_dir = dataset_config.mask_dir
        image_suffixes = dataset_config.image_suffixes
        mask_suffixes = dataset_config.mask_suffixes
    else:
        if not args.image_dir or not args.mask_dir:
            raise ValueError("You must provide image and mask directories.")
        image_dir = Path(args.image_dir)
        mask_dir = Path(args.mask_dir)
        image_suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        mask_suffixes = (".png", ".bmp", ".tif", ".tiff")
    return image_dir, mask_dir, image_suffixes, mask_suffixes


def main() -> None:
    args = parse_args()
    image_dir, mask_dir, image_suffixes, mask_suffixes = resolve_inputs(args)

    label_config = load_label_config(args.label_config) if args.label_config else None
    class_names = label_config.class_names if label_config else {}
    allowed_label_ids = set(label_config.class_ids) if label_config else None
    ignore_index = label_config.ignore_index if label_config else None
    background_ids = label_config.background_ids if label_config else (0,)

    scan_result = scan_dataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_suffixes=image_suffixes,
        mask_suffixes=mask_suffixes,
        allowed_label_ids=allowed_label_ids,
        ignore_index=ignore_index,
        background_ids=background_ids,
    )
    dataset_stats = compute_class_statistics(
        scan_result.matched_pairs,
        class_names=class_names,
        ignore_index=ignore_index,
        background_ids=background_ids,
    )

    output_dir = Path(args.output_dir)
    report_payload = {
        "scan": scan_result.to_dict(),
        "stats": dataset_stats.to_dict(),
    }
    write_json(output_dir / "summary.json", report_payload)
    write_csv(output_dir / "class_distribution.csv", [row.to_dict() for row in dataset_stats.class_rows])
    write_markdown(
        output_dir / "report.md",
        "Dataset Health Check",
        [
            f"- images: {scan_result.total_images}",
            f"- masks: {scan_result.total_masks}",
            f"- matched_pairs: {len(scan_result.matched_pairs)}",
            f"- missing_masks: {len(scan_result.missing_masks)}",
            f"- orphan_masks: {len(scan_result.orphan_masks)}",
            f"- size_mismatches: {len(scan_result.mismatched_shapes)}",
            f"- empty_masks: {len(scan_result.empty_masks)}",
            f"- invalid_label_samples: {len(scan_result.invalid_label_samples)}",
            "",
            "## Class Distribution",
            *[
                (
                    f"- {row.class_name} (id={row.class_id}): "
                    f"pixels={row.pixel_count}, samples={row.sample_count}"
                )
                for row in dataset_stats.class_rows
            ],
        ],
    )

    print(f"Matched pairs: {len(scan_result.matched_pairs)}")
    print(f"Missing masks: {len(scan_result.missing_masks)}")
    print(f"Orphan masks: {len(scan_result.orphan_masks)}")
    print(f"Size mismatches: {len(scan_result.mismatched_shapes)}")
    print(f"Empty masks: {len(scan_result.empty_masks)}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
