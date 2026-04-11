from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.common import setup_logger
from src.reporting.exporter import write_csv, write_json, write_markdown


MODEL_RUNS = [
    {"name": "UperNet_ConvNext_T_VOC_adv", "checkpoint": "models/UperNet_ConvNext_T_VOC_adv.pth"},
    {"name": "UperNet_ConvNext_T_VOC_clean", "checkpoint": "models/UperNet_ConvNext_T_VOC_clean.pth"},
    {"name": "UperNet_ResNet50_VOC_adv", "checkpoint": "models/UperNet_ResNet50_VOC_adv.pth"},
    {"name": "UperNet_ResNet50_VOC_clean", "checkpoint": "models/UperNet_ResNet50_VOC_clean.pth"},
    {"name": "Segmenter_ViT_S_VOC_adv", "checkpoint": "models/Segmenter_ViT_S_VOC_adv.pth"},
    {"name": "Segmenter_ViT_S_VOC_clean", "checkpoint": "models/Segmenter_ViT_S_VOC_clean.pth"},
]

RENAMES = {
    "UperNet_ConvNext_T_VOC": "UperNet_ConvNext_T_VOC_adv",
    "Segmenter_ViT_S_VOC": "Segmenter_ViT_S_VOC_adv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize VOC clean evaluation naming and rebuild aggregated summary.")
    parser.add_argument("--output-dir", default="results/reports/voc_clean_eval", help="Base output directory.")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    logger = setup_logger("voc_clean_eval.refresh", output_root / "refresh.log")

    for old_name, new_name in RENAMES.items():
        old_path = output_root / old_name
        new_path = output_root / new_name
        if old_path.exists() and not new_path.exists():
            old_path.rename(new_path)
            logger.info("Renamed %s -> %s", old_path, new_path)

    aggregated_rows: list[dict[str, object]] = []
    for item in MODEL_RUNS:
        run_output_dir = output_root / item["name"]
        summary_path = run_output_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary for {item['name']}: {summary_path}")
        summary = _load_json(summary_path)
        aggregated_rows.append(
            {
                "name": item["name"],
                "family": summary["model"]["family"],
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
        "output_dir": str(output_root.resolve()),
        "models": aggregated_rows,
    }
    write_json(output_root / "summary_all.json", report_payload)
    write_csv(output_root / "summary_all.csv", aggregated_rows)
    write_markdown(
        output_root / "summary_all.md",
        "VOC Clean Evaluation Summary",
        [
            f"- output_dir: {output_root.resolve()}",
            f"- total_models: {len(aggregated_rows)}",
            "",
            "## Results",
            *[
                f"- {row['name']}: mIoU={row['mIoU_percent']:.2f}, mAcc={row['mAcc_percent']:.2f}, aAcc={row['aAcc_percent']:.2f}"
                for row in aggregated_rows
            ],
        ],
    )
    logger.info("Refreshed aggregated summary at %s", output_root.resolve())
    print(f"Refreshed summary at: {output_root}")


if __name__ == "__main__":
    main()
