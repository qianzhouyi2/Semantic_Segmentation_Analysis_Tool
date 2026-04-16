from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401

from src.attacks import AttackConfig, AttackRunner
from src.common import setup_logger
from src.common.config import load_yaml
from src.datasets import PASCAL_VOC_CLASS_NAMES, PascalVOCValidationDataset
from src.metrics.segmentation import compute_confusion_matrix, summarize_confusion_matrix
from src.models import MODEL_FAMILY_CHOICES, TorchSegmentationModelAdapter, build_model_from_checkpoint
from src.reporting.exporter import write_csv, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate transfer-based black-box attacks for segmentation models.")
    parser.add_argument("--attack-config", required=True, help="Attack YAML config used on the source model.")
    parser.add_argument("--source-family", required=True, choices=MODEL_FAMILY_CHOICES, help="Source model family.")
    parser.add_argument("--source-checkpoint", required=True, help="Source checkpoint.")
    parser.add_argument("--source-defense-config", default="", help="Optional sparse defense config for source model.")
    parser.add_argument("--target-family", required=True, choices=MODEL_FAMILY_CHOICES, help="Target model family.")
    parser.add_argument("--target-checkpoint", required=True, help="Target checkpoint.")
    parser.add_argument("--target-defense-config", default="", help="Optional sparse defense config for target model.")
    parser.add_argument("--dataset-root", default="datasets", help="VOC dataset root.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=1, help="Transfer evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--num-classes", type=int, default=21, help="Segmentation class count.")
    parser.add_argument("--max-batches", type=int, default=-1, help="Optional early stop for debugging.")
    parser.add_argument("--strict", dest="strict", action="store_true", help="Strict checkpoint loading.")
    parser.add_argument("--no-strict", dest="strict", action="store_false", help="Allow checkpoint mismatch.")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        f"voc_transfer_eval.{Path(args.target_checkpoint).stem}.{Path(args.attack_config).stem}",
        output_dir / "evaluate.log",
    )
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    attack_config = AttackConfig.from_dict(load_yaml(args.attack_config))

    logger.info(
        "Starting transfer evaluation: source=%s target=%s attack=%s device=%s",
        Path(args.source_checkpoint).resolve(),
        Path(args.target_checkpoint).resolve(),
        attack_config.name,
        device,
    )

    dataset = PascalVOCValidationDataset(args.dataset_root, split="val", resize_short=473, crop_size=473)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    source_model, source_missing, source_unexpected = build_model_from_checkpoint(
        family=args.source_family,
        checkpoint_path=args.source_checkpoint,
        num_classes=args.num_classes,
        map_location="cpu",
        strict=args.strict,
        defense_config_path=args.source_defense_config or None,
    )
    target_model, target_missing, target_unexpected = build_model_from_checkpoint(
        family=args.target_family,
        checkpoint_path=args.target_checkpoint,
        num_classes=args.num_classes,
        map_location="cpu",
        strict=args.strict,
        defense_config_path=args.target_defense_config or None,
    )
    source_adapter = TorchSegmentationModelAdapter(model=source_model, num_classes=args.num_classes, device=device)
    target_adapter = TorchSegmentationModelAdapter(model=target_model, num_classes=args.num_classes, device=device)
    attack_runner = AttackRunner(source_adapter)

    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    linf_values: list[torch.Tensor] = []
    l2_values: list[torch.Tensor] = []
    filenames: list[str] = []
    processed_batches = 0
    processed_samples = 0

    for batch_index, batch in enumerate(dataloader):
        images = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        targets_cpu = targets.cpu().numpy()
        if len(batch) >= 3:
            filenames.extend([str(item) for item in batch[2]])

        attack_output = attack_runner.run(config=attack_config, images=images, targets=targets)
        with torch.no_grad():
            predictions = target_adapter.predict(attack_output.adversarial_images).cpu().numpy()

        perturbation = attack_output.perturbation.detach()
        linf_values.append(perturbation.abs().flatten(1).amax(dim=1).cpu())
        l2_values.append(perturbation.flatten(1).norm(p=2, dim=1).cpu())

        for target, prediction in zip(targets_cpu, predictions, strict=True):
            confusion += compute_confusion_matrix(
                target=target,
                prediction=prediction,
                num_classes=args.num_classes,
                ignore_index=None,
            )

        processed_batches += 1
        processed_samples += images.shape[0]
        if processed_batches == 1 or processed_batches % 20 == 0:
            logger.info("Transfer evaluation progress: batches=%d samples=%d", processed_batches, processed_samples)
        if args.max_batches > 0 and batch_index + 1 >= args.max_batches:
            break

    metrics = summarize_confusion_matrix(confusion, class_names=PASCAL_VOC_CLASS_NAMES)
    linf_tensor = torch.cat(linf_values) if linf_values else torch.empty(0)
    l2_tensor = torch.cat(l2_values) if l2_values else torch.empty(0)
    payload = {
        "source_model": {
            "family": args.source_family,
            "checkpoint": str(Path(args.source_checkpoint).resolve()),
            "defense_config": str(Path(args.source_defense_config).resolve()) if args.source_defense_config else None,
            "sparse_defense": getattr(source_model, "_sparse_defense_info", None),
            "missing_keys": source_missing,
            "unexpected_keys": source_unexpected,
        },
        "target_model": {
            "family": args.target_family,
            "checkpoint": str(Path(args.target_checkpoint).resolve()),
            "defense_config": str(Path(args.target_defense_config).resolve()) if args.target_defense_config else None,
            "sparse_defense": getattr(target_model, "_sparse_defense_info", None),
            "missing_keys": target_missing,
            "unexpected_keys": target_unexpected,
        },
        "dataset": {
            "root": str(Path(args.dataset_root).resolve()),
            "split": "val",
            "num_samples": len(dataset),
        },
        "processed_batches": processed_batches,
        "processed_samples": processed_samples,
        "filenames": filenames,
        "metrics": metrics.to_dict(),
        "reference": {
            "mAcc": float(metrics.mean_recall),
            "aAcc": float(metrics.pixel_accuracy),
            "mIoU": float(metrics.mean_iou),
        },
        "reference_percent": {
            "mAcc": float(metrics.mean_recall * 100.0),
            "aAcc": float(metrics.pixel_accuracy * 100.0),
            "mIoU": float(metrics.mean_iou * 100.0),
        },
        "attack": {
            "name": attack_config.name,
            "epsilon": attack_config.epsilon,
            "step_size": attack_config.resolved_step_size(),
            "steps": attack_config.steps,
            "random_start": attack_config.random_start,
            "targeted": attack_config.targeted,
            "loss_name": attack_config.loss_name,
            "ignore_index": attack_config.ignore_index,
            "mean_linf": float(linf_tensor.mean().item()) if linf_tensor.numel() else 0.0,
            "max_linf": float(linf_tensor.max().item()) if linf_tensor.numel() else 0.0,
            "mean_l2": float(l2_tensor.mean().item()) if l2_tensor.numel() else 0.0,
            "mode": "transfer_blackbox",
        },
    }

    write_json(output_dir / "summary.json", payload)
    write_csv(output_dir / "per_class_metrics.csv", payload["metrics"]["per_class"])
    write_markdown(
        output_dir / "report.md",
        "VOC Transfer Evaluation",
        [
            f"- source_family: {args.source_family}",
            f"- source_checkpoint: {Path(args.source_checkpoint).resolve()}",
            (
                f"- source_defense_config: {Path(args.source_defense_config).resolve()}"
                if args.source_defense_config
                else "- source_defense_config: <none>"
            ),
            f"- target_family: {args.target_family}",
            f"- target_checkpoint: {Path(args.target_checkpoint).resolve()}",
            (
                f"- target_defense_config: {Path(args.target_defense_config).resolve()}"
                if args.target_defense_config
                else "- target_defense_config: <none>"
            ),
            f"- attack: {attack_config.name}",
            f"- processed_samples: {processed_samples}",
            "",
            "## Metrics",
            f"- mIoU: {payload['reference_percent']['mIoU']:.2f}",
            f"- mAcc: {payload['reference_percent']['mAcc']:.2f}",
            f"- aAcc: {payload['reference_percent']['aAcc']:.2f}",
        ],
    )
    logger.info("Transfer evaluation complete: mIoU=%.2f output_dir=%s", payload["reference_percent"]["mIoU"], output_dir.resolve())
    print(f"Transfer evaluation written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
