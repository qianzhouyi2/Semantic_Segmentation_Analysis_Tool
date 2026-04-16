from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401

from src.common import setup_logger
from src.models import build_model_from_checkpoint
from src.models.sparse import (
    SUPPORTED_SPARSE_FAMILIES,
    apply_sparse_defense,
    calibrate_sparse_defense,
    export_sparse_sidecar,
    load_sparse_defense_config,
)
from src.datasets import PascalVOCValidationDataset
from src.reporting.exporter import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate and export sparse-defense statistics sidecars.")
    parser.add_argument(
        "--family",
        required=True,
        choices=sorted(SUPPORTED_SPARSE_FAMILIES),
        help="Model family to instantiate.",
    )
    parser.add_argument("--checkpoint", required=True, help="Base checkpoint path.")
    parser.add_argument("--defense-config", required=True, help="Sparse defense YAML config path.")
    parser.add_argument("--dataset-root", default="datasets", help="VOC dataset root that contains VOCdevkit/.")
    parser.add_argument("--dataset-split", default="val", help="VOC split file name under ImageSets/Segmentation.")
    parser.add_argument("--output-stats", default="", help="Override the stats sidecar path from the defense config.")
    parser.add_argument("--summary-path", default="", help="Optional calibration summary path.")
    parser.add_argument("--batch-size", type=int, default=4, help="Calibration batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument("--num-classes", type=int, default=21, help="Segmentation class count.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Optional cap on calibration samples.")
    parser.add_argument("--strict", dest="strict", action="store_true", help="Require exact checkpoint key match.")
    parser.add_argument("--no-strict", dest="strict", action="store_false", help="Allow checkpoint key mismatch.")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def resolve_output_stats(args: argparse.Namespace, config_path: Path) -> Path:
    if args.output_stats:
        return Path(args.output_stats)
    defense_config = load_sparse_defense_config(config_path)
    if defense_config.stats_path is None:
        raise ValueError("Sparse defense config does not define `stats_path`; pass `--output-stats`.")
    return defense_config.stats_path


def resolve_summary_path(args: argparse.Namespace, output_stats: Path) -> Path:
    if args.summary_path:
        return Path(args.summary_path)
    return output_stats.with_suffix(".summary.json")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    defense_config_path = Path(args.defense_config)
    defense_config = load_sparse_defense_config(defense_config_path)
    output_stats = resolve_output_stats(args, defense_config_path)
    summary_path = resolve_summary_path(args, output_stats)
    output_stats.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        f"sparse_prepare.{checkpoint_path.stem}.{defense_config.variant}",
        output_stats.parent / "prepare_sparse_defense.log",
    )
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    logger.info("Loading base checkpoint")
    logger.info(
        "family=%s checkpoint=%s defense=%s device=%s split=%s",
        args.family,
        checkpoint_path.resolve(),
        defense_config.variant,
        device,
        args.dataset_split,
    )

    model, missing_keys, unexpected_keys = build_model_from_checkpoint(
        family=args.family,
        checkpoint_path=checkpoint_path,
        num_classes=args.num_classes,
        map_location="cpu",
        strict=args.strict,
    )
    apply_sparse_defense(model, family=args.family, config=defense_config, load_stats=False)
    model.to(device)
    model.eval()

    dataset = PascalVOCValidationDataset(
        args.dataset_root,
        split=args.dataset_split,
        resize_short=473,
        crop_size=473,
        remap_ignore_to_background=False,
    )
    if args.max_samples > 0:
        dataset.sample_ids = dataset.sample_ids[: args.max_samples]
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    calibration_info = calibrate_sparse_defense(
        model,
        dataloader,
        config=defense_config,
        ignore_index=255,
    )
    export_payload = export_sparse_sidecar(
        model,
        family=args.family,
        config=defense_config,
        output_path=output_stats,
        metadata={
            "checkpoint": str(checkpoint_path.resolve()),
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "dataset_split": args.dataset_split,
            "num_samples": len(dataset),
        },
    )

    summary = {
        "model": {
            "family": args.family,
            "checkpoint": str(checkpoint_path.resolve()),
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        },
        "defense": {
            "config_path": str(defense_config_path.resolve()),
            "variant": defense_config.variant,
            "stats_path": str(output_stats.resolve()),
        },
        "dataset": {
            "root": str(Path(args.dataset_root).resolve()),
            "split": args.dataset_split,
            "num_samples": len(dataset),
        },
        "calibration": calibration_info,
        "sidecar": {
            "format_version": export_payload["format_version"],
            "num_sparse_modules": export_payload["num_sparse_modules"],
            "path": str(output_stats.resolve()),
        },
    }
    write_json(summary_path, summary)

    logger.info(
        "Sparse defense calibration complete: variant=%s modules=%d stats=%s summary=%s",
        defense_config.variant,
        export_payload["num_sparse_modules"],
        output_stats.resolve(),
        summary_path.resolve(),
    )
    print(f"Sparse stats written to: {output_stats.resolve()}")
    print(f"Calibration summary written to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
