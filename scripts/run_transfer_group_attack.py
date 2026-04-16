from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401

from src.attacks import AttackConfig, AttackRunner
from src.common import setup_logger
from src.datasets import PASCAL_VOC_CLASS_NAMES, PascalVOCValidationDataset
from src.metrics.segmentation import compute_confusion_matrix, summarize_confusion_matrix
from src.models import TorchSegmentationModelAdapter, build_model_from_checkpoint
from src.reporting.exporter import write_csv, write_json, write_markdown
from src.common.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict shared-input transfer evaluation for a source-target group.")
    parser.add_argument("--case-json", required=True, help="Path to one transfer case JSON file.")
    parser.add_argument("--dataset-root", default="datasets", help="VOC dataset root.")
    parser.add_argument("--output-dir", required=True, help="Output directory for this group case.")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--num-classes", type=int, default=21, help="Segmentation class count.")
    parser.add_argument("--max-batches", type=int, default=-1, help="Optional early stop for debugging.")
    parser.add_argument("--strict", dest="strict", action="store_true", help="Strict checkpoint loading.")
    parser.add_argument("--no-strict", dest="strict", action="store_false", help="Allow checkpoint mismatch.")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_adapter(
    *,
    family: str,
    checkpoint: str,
    defense_config: str | None,
    num_classes: int,
    strict: bool,
    device: torch.device,
) -> tuple[TorchSegmentationModelAdapter, dict]:
    model, missing_keys, unexpected_keys = build_model_from_checkpoint(
        family=family,
        checkpoint_path=checkpoint,
        num_classes=num_classes,
        map_location="cpu",
        strict=strict,
        defense_config_path=defense_config or None,
    )
    adapter = TorchSegmentationModelAdapter(model=model, num_classes=num_classes, device=device)
    return adapter, {
        "family": family,
        "checkpoint": str(Path(checkpoint).resolve()),
        "defense_config": None if defense_config is None else str(Path(defense_config).resolve()),
        "sparse_defense": getattr(model, "_sparse_defense_info", None),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


def init_confusion(num_classes: int) -> np.ndarray:
    return np.zeros((num_classes, num_classes), dtype=np.int64)


def update_confusion(confusion: np.ndarray, targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> None:
    for target, prediction in zip(targets, predictions, strict=True):
        confusion += compute_confusion_matrix(
            target=target,
            prediction=prediction,
            num_classes=num_classes,
            ignore_index=None,
        )


def summarize_metrics(confusion: np.ndarray) -> dict:
    metrics = summarize_confusion_matrix(confusion, class_names=PASCAL_VOC_CLASS_NAMES)
    return {
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
    }


def main() -> None:
    args = parse_args()
    case = load_json(Path(args.case_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        f"transfer_group.{case['case_id']}",
        output_dir / "evaluate.log",
    )
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    attack_config = AttackConfig.from_dict(load_yaml(Path(case["attack_config"])))

    logger.info(
        "Starting transfer group evaluation: case=%s attack=%s source=%s device=%s",
        case["case_id"],
        case["attack_stem"],
        case["source"]["model_id"],
        device,
    )

    source_adapter, source_info = build_adapter(
        family=case["source"]["family"],
        checkpoint=case["source"]["checkpoint"],
        defense_config=case["source"]["defense_config"],
        num_classes=args.num_classes,
        strict=args.strict,
        device=device,
    )
    target_adapters: dict[str, TorchSegmentationModelAdapter] = {}
    target_infos: dict[str, dict] = {}
    for target in case["targets"]:
        adapter, info = build_adapter(
            family=target["family"],
            checkpoint=target["checkpoint"],
            defense_config=target["defense_config"],
            num_classes=args.num_classes,
            strict=args.strict,
            device=device,
        )
        target_adapters[target["model_id"]] = adapter
        target_infos[target["model_id"]] = info

    dataset = PascalVOCValidationDataset(args.dataset_root, split="val", resize_short=473, crop_size=473)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    attack_runner = AttackRunner(source_adapter)

    clean_confusions: dict[str, np.ndarray] = {"source_self": init_confusion(args.num_classes)}
    adv_confusions: dict[str, np.ndarray] = {"source_self": init_confusion(args.num_classes)}
    for target in case["targets"]:
        clean_confusions[target["model_id"]] = init_confusion(args.num_classes)
        adv_confusions[target["model_id"]] = init_confusion(args.num_classes)

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

        with torch.no_grad():
            source_clean_pred = source_adapter.predict(images).cpu().numpy()
        update_confusion(clean_confusions["source_self"], targets_cpu, source_clean_pred, args.num_classes)

        for target in case["targets"]:
            target_id = target["model_id"]
            with torch.no_grad():
                clean_pred = target_adapters[target_id].predict(images).cpu().numpy()
            update_confusion(clean_confusions[target_id], targets_cpu, clean_pred, args.num_classes)

        attack_output = attack_runner.run(config=attack_config, images=images, targets=targets)
        with torch.no_grad():
            source_adv_pred = source_adapter.predict(attack_output.adversarial_images).cpu().numpy()
        update_confusion(adv_confusions["source_self"], targets_cpu, source_adv_pred, args.num_classes)

        for target in case["targets"]:
            target_id = target["model_id"]
            with torch.no_grad():
                adv_pred = target_adapters[target_id].predict(attack_output.adversarial_images).cpu().numpy()
            update_confusion(adv_confusions[target_id], targets_cpu, adv_pred, args.num_classes)

        perturbation = attack_output.perturbation.detach()
        linf_values.append(perturbation.abs().flatten(1).amax(dim=1).cpu())
        l2_values.append(perturbation.flatten(1).norm(p=2, dim=1).cpu())

        processed_batches += 1
        processed_samples += images.shape[0]
        if processed_batches == 1 or processed_batches % 20 == 0:
            logger.info("Transfer group progress: batches=%d samples=%d", processed_batches, processed_samples)
        if args.max_batches > 0 and batch_index + 1 >= args.max_batches:
            break

    linf_tensor = torch.cat(linf_values) if linf_values else torch.empty(0)
    l2_tensor = torch.cat(l2_values) if l2_values else torch.empty(0)

    source_self = {
        "model": source_info | {"model_id": case["source"]["model_id"]},
        "clean": summarize_metrics(clean_confusions["source_self"]),
        "transfer": summarize_metrics(adv_confusions["source_self"]),
    }
    source_self["transfer_miou_drop"] = (
        source_self["clean"]["reference_percent"]["mIoU"] - source_self["transfer"]["reference_percent"]["mIoU"]
    )

    target_payloads: list[dict] = []
    for target in case["targets"]:
        target_id = target["model_id"]
        clean_summary = summarize_metrics(clean_confusions[target_id])
        transfer_summary = summarize_metrics(adv_confusions[target_id])
        target_payloads.append(
            {
                "model": target_infos[target_id] | {"model_id": target_id},
                "clean": clean_summary,
                "transfer": transfer_summary,
                "transfer_miou_drop": clean_summary["reference_percent"]["mIoU"] - transfer_summary["reference_percent"]["mIoU"],
            }
        )

    payload = {
        "case_id": case["case_id"],
        "regime": case["regime"],
        "relation": case["relation"],
        "attack": {
            "stem": case["attack_stem"],
            "name": attack_config.name,
            "config": str(Path(case["attack_config"]).resolve()),
            "epsilon": attack_config.epsilon,
            "step_size": attack_config.resolved_step_size(),
            "steps": attack_config.steps,
            "random_start": attack_config.random_start,
        },
        "dataset": {
            "root": str(Path(args.dataset_root).resolve()),
            "split": "val",
            "num_samples": len(dataset),
        },
        "processed_batches": processed_batches,
        "processed_samples": processed_samples,
        "filenames": filenames,
        "perturbation": {
            "mean_linf": float(linf_tensor.mean().item()) if linf_tensor.numel() else 0.0,
            "max_linf": float(linf_tensor.max().item()) if linf_tensor.numel() else 0.0,
            "mean_l2": float(l2_tensor.mean().item()) if l2_tensor.numel() else 0.0,
        },
        "source_self": source_self,
        "targets": target_payloads,
    }

    write_json(output_dir / "summary.json", payload)
    write_markdown(
        output_dir / "report.md",
        "VOC Transfer Group Evaluation",
        [
            f"- case_id: {case['case_id']}",
            f"- relation: {case['relation']}",
            f"- regime: {case['regime']}",
            f"- attack: {attack_config.name}",
            f"- source: {case['source']['model_id']}",
            f"- processed_samples: {processed_samples}",
            "",
            "## Source Self",
            f"- clean mIoU: {source_self['clean']['reference_percent']['mIoU']:.2f}",
            f"- transfer mIoU: {source_self['transfer']['reference_percent']['mIoU']:.2f}",
            f"- transfer drop: {source_self['transfer_miou_drop']:.2f}",
            "",
            "## Targets",
            *[
                (
                    f"- {item['model']['model_id']}: clean={item['clean']['reference_percent']['mIoU']:.2f}, "
                    f"transfer={item['transfer']['reference_percent']['mIoU']:.2f}, "
                    f"drop={item['transfer_miou_drop']:.2f}"
                )
                for item in target_payloads
            ],
        ],
    )
    target_rows = []
    for item in target_payloads:
        target_rows.append(
            {
                "model_id": item["model"]["model_id"],
                "family": item["model"]["family"],
                "variant": item["model"]["sparse_defense"]["variant"] if item["model"]["sparse_defense"] else "baseline",
                "clean_miou": item["clean"]["reference_percent"]["mIoU"],
                "transfer_miou": item["transfer"]["reference_percent"]["mIoU"],
                "transfer_miou_drop": item["transfer_miou_drop"],
            }
        )
    write_csv(output_dir / "target_metrics.csv", target_rows)
    logger.info("Transfer group evaluation complete: case=%s output_dir=%s", case["case_id"], output_dir.resolve())
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
