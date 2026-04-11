from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401

from src.attacks import AttackConfig
from src.common import setup_logger
from src.common.config import load_yaml
from src.datasets import PASCAL_VOC_CLASS_NAMES, PascalVOCValidationDataset
from src.evaluation import evaluate_adversarial_segmentation_model
from src.models import MODEL_FAMILY_CHOICES, TorchSegmentationModelAdapter, build_model_from_checkpoint
from src.reporting.exporter import write_csv, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint under adversarial attack.")
    parser.add_argument("--attack-config", required=True, help="Path to the attack YAML config.")
    parser.add_argument("--family", required=True, choices=MODEL_FAMILY_CHOICES, help="Model family to instantiate.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--dataset-root", default="datasets", help="VOC dataset root that contains VOCdevkit/.")
    parser.add_argument("--output-dir", default="", help="Directory for outputs. Defaults to results/reports/voc_adv_eval/<checkpoint>_<attack>.")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument("--num-classes", type=int, default=21, help="Segmentation class count.")
    parser.add_argument("--max-batches", type=int, default=-1, help="Optional early-stop for debugging.")
    parser.add_argument("--strict", dest="strict", action="store_true", help="Require exact checkpoint key match.")
    parser.add_argument("--no-strict", dest="strict", action="store_false", help="Allow missing or unexpected checkpoint keys.")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace, attack_name: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    checkpoint_stem = Path(args.checkpoint).stem
    return Path("results/reports/voc_adv_eval") / f"{checkpoint_stem}_{attack_name}"


def main() -> None:
    args = parse_args()
    attack_config = AttackConfig.from_dict(load_yaml(args.attack_config))
    checkpoint_path = Path(args.checkpoint)
    output_dir = resolve_output_dir(args, attack_config.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"voc_adv_eval.{checkpoint_path.stem}.{attack_config.name}", output_dir / "evaluate.log")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    logger.info("Starting adversarial evaluation")
    logger.info(
        "family=%s checkpoint=%s device=%s attack=%s epsilon=%s step_size=%s steps=%s",
        args.family,
        checkpoint_path.resolve(),
        device,
        attack_config.name,
        attack_config.epsilon,
        attack_config.resolved_step_size(),
        attack_config.steps,
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
    )
    adapter = TorchSegmentationModelAdapter(model=model, num_classes=args.num_classes, device=device)
    logger.info("Checkpoint loaded: missing_keys=%d unexpected_keys=%d", len(missing_keys), len(unexpected_keys))

    summary = evaluate_adversarial_segmentation_model(
        model=adapter,
        attack_config=attack_config,
        dataloader=dataloader,
        ignore_index=None,
        class_names=PASCAL_VOC_CLASS_NAMES,
        max_batches=args.max_batches,
        logger=logger,
    )
    payload = {
        "model": {
            "family": args.family,
            "checkpoint": str(checkpoint_path.resolve()),
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
        "VOC Adversarial Evaluation",
        [
            f"- family: {args.family}",
            f"- checkpoint: {checkpoint_path.resolve()}",
            f"- dataset_root: {Path(args.dataset_root).resolve()}",
            f"- attack: {attack_config.name}",
            f"- epsilon: {attack_config.epsilon}",
            f"- step_size: {attack_config.resolved_step_size()}",
            f"- steps: {attack_config.steps}",
            f"- random_start: {attack_config.random_start}",
            f"- targeted: {attack_config.targeted}",
            f"- processed_samples: {payload['processed_samples']}",
            f"- processed_batches: {payload['processed_batches']}",
            "",
            "## Reference Metrics",
            f"- mIoU: {payload['reference_percent']['mIoU']:.2f}",
            f"- mAcc: {payload['reference_percent']['mAcc']:.2f}",
            f"- aAcc: {payload['reference_percent']['aAcc']:.2f}",
            "",
            "## Perturbation",
            f"- mean_linf: {payload['attack']['mean_linf']:.6f}",
            f"- max_linf: {payload['attack']['max_linf']:.6f}",
            f"- mean_l2: {payload['attack']['mean_l2']:.6f}",
        ],
    )

    logger.info(
        "Finished adversarial evaluation: mIoU=%.2f mAcc=%.2f aAcc=%.2f output_dir=%s",
        payload["reference_percent"]["mIoU"],
        payload["reference_percent"]["mAcc"],
        payload["reference_percent"]["aAcc"],
        output_dir.resolve(),
    )
    print(
        f"{checkpoint_path.name}: attack={attack_config.name} "
        f"mIoU={payload['reference_percent']['mIoU']:.2f} "
        f"mAcc={payload['reference_percent']['mAcc']:.2f} "
        f"aAcc={payload['reference_percent']['aAcc']:.2f}"
    )
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
