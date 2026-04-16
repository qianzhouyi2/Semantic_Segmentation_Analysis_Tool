from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401

from src.common import setup_logger
from src.datasets import PASCAL_VOC_CLASS_NAMES, PascalVOCValidationDataset
from src.evaluation import evaluate_segmentation_model
from src.models import MODEL_FAMILY_CHOICES, build_model_from_checkpoint
from src.reporting.exporter import write_csv, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint on Pascal VOC clean validation data.")
    parser.add_argument("--family", required=True, choices=MODEL_FAMILY_CHOICES, help="Model family to instantiate.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--defense-config", default="", help="Optional sparse defense YAML config.")
    parser.add_argument("--dataset-root", default="datasets", help="VOC dataset root that contains VOCdevkit/.")
    parser.add_argument("--output-dir", default="", help="Directory for outputs. Defaults to results/reports/voc_clean_eval/<checkpoint_stem>.")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument("--num-classes", type=int, default=21, help="Segmentation class count.")
    parser.add_argument("--max-batches", type=int, default=-1, help="Optional early-stop for debugging.")
    parser.add_argument("--strict", action="store_true", default=True, help="Require exact checkpoint key match.")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    checkpoint_stem = Path(args.checkpoint).stem
    return Path("results/reports/voc_clean_eval") / checkpoint_stem


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"voc_clean_eval.{checkpoint_path.stem}", output_dir / "evaluate.log")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    logger.info("Starting VOC clean evaluation")
    logger.info(
        "family=%s checkpoint=%s device=%s defense_config=%s",
        args.family,
        checkpoint_path.resolve(),
        device,
        Path(args.defense_config).resolve() if args.defense_config else "<none>",
    )

    dataset = PascalVOCValidationDataset(args.dataset_root, split="val", resize_short=473, crop_size=473)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model, missing_keys, unexpected_keys = build_model_from_checkpoint(
        family=args.family,
        checkpoint_path=checkpoint_path,
        num_classes=args.num_classes,
        map_location="cpu",
        strict=args.strict,
        defense_config_path=args.defense_config or None,
    )
    logger.info("Checkpoint loaded: missing_keys=%d unexpected_keys=%d", len(missing_keys), len(unexpected_keys))
    sparse_defense_info = getattr(model, "_sparse_defense_info", None)
    summary = evaluate_segmentation_model(
        model=model,
        dataloader=dataloader,
        num_classes=args.num_classes,
        device=device,
        ignore_index=None,
        class_names=PASCAL_VOC_CLASS_NAMES,
        max_batches=args.max_batches,
        logger=logger,
    )

    payload = {
        "model": {
            "family": args.family,
            "checkpoint": str(checkpoint_path.resolve()),
            "defense_config": str(Path(args.defense_config).resolve()) if args.defense_config else None,
            "sparse_defense": sparse_defense_info,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        },
        "dataset": {
            "root": str(Path(args.dataset_root).resolve()),
            "split": "val",
            "resize_short": 473,
            "crop_size": 473,
            "num_samples": len(dataset),
        },
        **summary,
    }

    write_json(output_dir / "summary.json", payload)
    write_csv(output_dir / "per_class_metrics.csv", payload["metrics"]["per_class"])
    write_markdown(
        output_dir / "report.md",
        "VOC Clean Evaluation",
        [
            f"- family: {args.family}",
            f"- checkpoint: {checkpoint_path.resolve()}",
            (
                f"- defense_config: {Path(args.defense_config).resolve()}"
                if args.defense_config
                else "- defense_config: <none>"
            ),
            f"- dataset_root: {Path(args.dataset_root).resolve()}",
            f"- processed_samples: {payload['processed_samples']}",
            f"- processed_batches: {payload['processed_batches']}",
            "",
            "## Reference Metrics",
            f"- mIoU: {payload['reference_percent']['mIoU']:.2f}",
            f"- mAcc: {payload['reference_percent']['mAcc']:.2f}",
            f"- aAcc: {payload['reference_percent']['aAcc']:.2f}",
            "",
            "## Extended Metrics",
            f"- pixel_accuracy: {payload['metrics']['pixel_accuracy'] * 100.0:.2f}",
            f"- mean_dice: {payload['metrics']['mean_dice'] * 100.0:.2f}",
            f"- mean_precision: {payload['metrics']['mean_precision'] * 100.0:.2f}",
            f"- mean_recall: {payload['metrics']['mean_recall'] * 100.0:.2f}",
            f"- mean_f1: {payload['metrics']['mean_f1'] * 100.0:.2f}",
        ],
    )

    logger.info(
        "Finished evaluation: mIoU=%.2f mAcc=%.2f aAcc=%.2f output_dir=%s",
        payload["reference_percent"]["mIoU"],
        payload["reference_percent"]["mAcc"],
        payload["reference_percent"]["aAcc"],
        output_dir.resolve(),
    )
    print(
        f"{checkpoint_path.name}: mIoU={payload['reference_percent']['mIoU']:.2f} "
        f"mAcc={payload['reference_percent']['mAcc']:.2f} "
        f"aAcc={payload['reference_percent']['aAcc']:.2f}"
    )
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
