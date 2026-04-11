from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src.common import setup_logger
from src.reporting.exporter import write_csv, write_json, write_markdown


MODEL_RUNS = [
    {"name": "UperNet_ConvNext_T_VOC_adv", "family": "upernet_convnext", "checkpoint": "models/UperNet_ConvNext_T_VOC_adv.pth"},
    {"name": "UperNet_ConvNext_T_VOC_clean", "family": "upernet_convnext", "checkpoint": "models/UperNet_ConvNext_T_VOC_clean.pth"},
    {"name": "UperNet_ResNet50_VOC_adv", "family": "upernet_resnet50", "checkpoint": "models/UperNet_ResNet50_VOC_adv.pth"},
    {"name": "UperNet_ResNet50_VOC_clean", "family": "upernet_resnet50", "checkpoint": "models/UperNet_ResNet50_VOC_clean.pth"},
    {"name": "Segmenter_ViT_S_VOC_adv", "family": "segmenter_vit_s", "checkpoint": "models/Segmenter_ViT_S_VOC_adv.pth"},
    {"name": "Segmenter_ViT_S_VOC_clean", "family": "segmenter_vit_s", "checkpoint": "models/Segmenter_ViT_S_VOC_clean.pth"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VOC clean evaluation for all bundled checkpoints.")
    parser.add_argument("--dataset-root", default="datasets", help="VOC dataset root that contains VOCdevkit/.")
    parser.add_argument("--output-dir", default="results/reports/voc_clean_eval", help="Base output directory.")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size per run.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument("--max-batches", type=int, default=-1, help="Optional early-stop for debugging.")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("voc_clean_eval.batch", output_root / "batch.log")
    logger.info("Starting batch VOC clean evaluation")
    logger.info(
        "dataset_root=%s device=%s batch_size=%d num_workers=%d",
        Path(args.dataset_root).resolve(),
        args.device,
        args.batch_size,
        args.num_workers,
    )

    aggregated_rows: list[dict[str, object]] = []
    for item in MODEL_RUNS:
        run_output_dir = output_root / item["name"]
        logger.info("Launching model=%s family=%s checkpoint=%s", item["name"], item["family"], Path(item["checkpoint"]).resolve())
        cmd = [
            sys.executable,
            "scripts/evaluate_voc_clean.py",
            "--family",
            item["family"],
            "--checkpoint",
            item["checkpoint"],
            "--dataset-root",
            args.dataset_root,
            "--output-dir",
            str(run_output_dir),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--max-batches",
            str(args.max_batches),
        ]
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if completed.stdout.strip():
            logger.info("Subprocess stdout for %s:\n%s", item["name"], completed.stdout.strip())
        if completed.stderr.strip():
            logger.warning("Subprocess stderr for %s:\n%s", item["name"], completed.stderr.strip())

        summary = _load_json(run_output_dir / "summary.json")
        logger.info(
            "Completed model=%s mIoU=%.2f mAcc=%.2f aAcc=%.2f",
            item["name"],
            float(summary["reference_percent"]["mIoU"]),
            float(summary["reference_percent"]["mAcc"]),
            float(summary["reference_percent"]["aAcc"]),
        )
        aggregated_rows.append(
            {
                "name": item["name"],
                "family": item["family"],
                "checkpoint": str(Path(item["checkpoint"]).resolve()),
                "mIoU_percent": float(summary["reference_percent"]["mIoU"]),
                "mAcc_percent": float(summary["reference_percent"]["mAcc"]),
                "aAcc_percent": float(summary["reference_percent"]["aAcc"]),
                "pixel_accuracy_percent": float(summary["metrics"]["pixel_accuracy"] * 100.0),
                "mean_dice_percent": float(summary["metrics"]["mean_dice"] * 100.0),
                "processed_samples": int(summary["processed_samples"]),
                "output_dir": str(run_output_dir.resolve()),
            }
        )

    aggregated_rows.sort(key=lambda row: float(row["mIoU_percent"]), reverse=True)
    report_payload = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "output_dir": str(output_root.resolve()),
        "models": aggregated_rows,
    }
    write_json(output_root / "summary_all.json", report_payload)
    write_csv(output_root / "summary_all.csv", aggregated_rows)
    write_markdown(
        output_root / "summary_all.md",
        "VOC Clean Evaluation Summary",
        [
            f"- dataset_root: {Path(args.dataset_root).resolve()}",
            f"- total_models: {len(aggregated_rows)}",
            "",
            "## Results",
            *[
                f"- {row['name']}: mIoU={row['mIoU_percent']:.2f}, mAcc={row['mAcc_percent']:.2f}, aAcc={row['aAcc_percent']:.2f}"
                for row in aggregated_rows
            ],
        ],
    )
    logger.info("Batch evaluation finished. Summary written to %s", output_root.resolve())
    print(f"Aggregated summary written to: {output_root}")


if __name__ == "__main__":
    main()
