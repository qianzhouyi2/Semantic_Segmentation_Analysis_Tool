from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import DataLoader

from src.attacks import AttackConfig
from src.common import setup_logger
from src.common.config import load_yaml
from src.common.sample_manifest import normalize_voc_sample_id
from src.datasets import PASCAL_VOC_CLASS_NAMES, PascalVOCValidationDataset
from src.evaluation import evaluate_adversarial_segmentation_model, evaluate_segmentation_model
from src.models import MODEL_FAMILY_CHOICES, TorchSegmentationModelAdapter, build_model_from_checkpoint, load_sparse_defense_config
from src.reporting.exporter import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find VOC samples whose clean per-sample quality improves after applying a sparse defense."
    )
    parser.add_argument("--family", required=True, choices=MODEL_FAMILY_CHOICES, help="Model family.")
    parser.add_argument("--checkpoint", required=True, help="Base checkpoint path.")
    parser.add_argument("--defense-config", required=True, help="Sparse defense YAML config path.")
    parser.add_argument("--attack-config", default="", help="Optional attack YAML. When set, compare defended vs baseline under attack.")
    parser.add_argument("--dataset-root", default="datasets", help="VOC dataset root.")
    parser.add_argument("--dataset-split", default="val", help="VOC split file under ImageSets/Segmentation.")
    parser.add_argument("--metric", choices=("sample_miou", "pixel_accuracy", "sample_dice"), default="sample_miou")
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.05,
        help="Minimum defended-baseline improvement required for a sample to count as rescued.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Output JSON path. Defaults to samples/<checkpoint>_<variant>_rescued_samples.json.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--num-classes", type=int, default=21, help="Segmentation class count.")
    parser.add_argument("--max-batches", type=int, default=-1, help="Optional early stop for debugging.")
    parser.add_argument("--strict", dest="strict", action="store_true", help="Strict checkpoint loading.")
    parser.add_argument("--no-strict", dest="strict", action="store_false", help="Allow checkpoint mismatch.")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def resolve_output_json(args: argparse.Namespace) -> Path:
    if args.output_json:
        return Path(args.output_json)
    defense_config = load_sparse_defense_config(args.defense_config)
    checkpoint_stem = Path(args.checkpoint).stem
    if args.attack_config:
        attack_stem = Path(args.attack_config).stem
        return Path("samples") / f"{checkpoint_stem}_{defense_config.variant}_{attack_stem}_rescued_samples.json"
    return Path("samples") / f"{checkpoint_stem}_{defense_config.variant}_rescued_samples.json"


def build_dataloader(
    *,
    dataset_root: str,
    dataset_split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[PascalVOCValidationDataset, DataLoader]:
    dataset = PascalVOCValidationDataset(
        dataset_root,
        split=dataset_split,
        resize_short=473,
        crop_size=473,
        remap_ignore_to_background=False,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return dataset, dataloader


def collect_per_sample_rows(
    *,
    family: str,
    checkpoint_path: Path,
    defense_config_path: str | None,
    attack_config: AttackConfig | None,
    dataset_root: str,
    dataset_split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    num_classes: int,
    strict: bool,
    max_batches: int,
    logger,
) -> tuple[dict[str, dict[str, float | str]], dict | None]:
    dataset, dataloader = build_dataloader(
        dataset_root=dataset_root,
        dataset_split=dataset_split,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    model, _missing_keys, _unexpected_keys = build_model_from_checkpoint(
        family=family,
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        map_location="cpu",
        strict=strict,
        defense_config_path=defense_config_path,
    )
    if attack_config is None:
        summary = evaluate_segmentation_model(
            model=model,
            dataloader=dataloader,
            num_classes=num_classes,
            device=device,
            ignore_index=255,
            class_names=PASCAL_VOC_CLASS_NAMES,
            max_batches=max_batches,
            logger=logger,
            collect_per_sample=True,
        )
    else:
        adapter = TorchSegmentationModelAdapter(model=model, num_classes=num_classes, device=device)
        summary = evaluate_adversarial_segmentation_model(
            model=adapter,
            attack_config=attack_config,
            dataloader=dataloader,
            ignore_index=255,
            class_names=PASCAL_VOC_CLASS_NAMES,
            max_batches=max_batches,
            logger=logger,
            collect_per_sample=True,
        )
    sparse_info = getattr(model, "_sparse_defense_info", None)
    rows: dict[str, dict[str, float | str]] = {}
    for row in summary["per_sample_metrics"]:
        sample_id = normalize_voc_sample_id(str(row["filename"]))
        rows[sample_id] = {
            "filename": str(row["filename"]),
            "pixel_accuracy": float(row["pixel_accuracy"]),
            "sample_miou": float(row["sample_miou"]),
            "sample_dice": float(row["sample_dice"]),
            "valid_class_count": float(row["valid_class_count"]),
        }
    return rows, sparse_info


def find_rescued_rows(
    baseline_rows: dict[str, dict[str, float | str]],
    defended_rows: dict[str, dict[str, float | str]],
    *,
    metric: str,
    min_delta: float,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    rescued_rows: list[dict[str, float | str]] = []
    improved_rows: list[dict[str, float | str]] = []
    common_ids = sorted(set(baseline_rows) & set(defended_rows))
    for sample_id in common_ids:
        baseline_value = float(baseline_rows[sample_id][metric])
        defended_value = float(defended_rows[sample_id][metric])
        delta = defended_value - baseline_value
        row = {
            "sample_id": sample_id,
            "filename": str(defended_rows[sample_id]["filename"]),
            f"baseline_{metric}": baseline_value,
            f"defended_{metric}": defended_value,
            "delta": delta,
            "baseline_pixel_accuracy": float(baseline_rows[sample_id]["pixel_accuracy"]),
            "defended_pixel_accuracy": float(defended_rows[sample_id]["pixel_accuracy"]),
            "baseline_sample_dice": float(baseline_rows[sample_id]["sample_dice"]),
            "defended_sample_dice": float(defended_rows[sample_id]["sample_dice"]),
        }
        if delta > 0:
            improved_rows.append(row)
        if delta >= min_delta:
            rescued_rows.append(row)
    rescued_rows.sort(key=lambda row: (-float(row["delta"]), str(row["sample_id"])))
    improved_rows.sort(key=lambda row: (-float(row["delta"]), str(row["sample_id"])))
    return rescued_rows, improved_rows


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    defense_config_path = Path(args.defense_config)
    defense_config = load_sparse_defense_config(defense_config_path)
    output_json = resolve_output_json(args)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    attack_config = None if not args.attack_config else AttackConfig.from_dict(load_yaml(args.attack_config))
    log_stem = output_json.stem
    logger = setup_logger(
        f"rescued_samples.{checkpoint_path.stem}.{defense_config.variant}",
        Path("logs") / f"{log_stem}.log",
    )
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    logger.info(
        "Starting rescued sample search: family=%s checkpoint=%s defense=%s attack=%s metric=%s min_delta=%.4f split=%s device=%s",
        args.family,
        checkpoint_path.resolve(),
        defense_config_path.resolve(),
        "<none>" if attack_config is None else Path(args.attack_config).resolve(),
        args.metric,
        args.min_delta,
        args.dataset_split,
        device,
    )

    logger.info("Collecting baseline per-sample metrics")
    baseline_rows, _ = collect_per_sample_rows(
        family=args.family,
        checkpoint_path=checkpoint_path,
        defense_config_path=None,
        attack_config=attack_config,
        dataset_root=args.dataset_root,
        dataset_split=args.dataset_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        num_classes=args.num_classes,
        strict=args.strict,
        max_batches=args.max_batches,
        logger=logger,
    )
    logger.info("Collecting defended per-sample metrics")
    defended_rows, sparse_info = collect_per_sample_rows(
        family=args.family,
        checkpoint_path=checkpoint_path,
        defense_config_path=str(defense_config_path),
        attack_config=attack_config,
        dataset_root=args.dataset_root,
        dataset_split=args.dataset_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        num_classes=args.num_classes,
        strict=args.strict,
        max_batches=args.max_batches,
        logger=logger,
    )

    rescued_rows, improved_rows = find_rescued_rows(
        baseline_rows,
        defended_rows,
        metric=args.metric,
        min_delta=args.min_delta,
    )

    payload = {
        "type": "voc_sample_id_list",
        "name": f"{checkpoint_path.stem}_{defense_config.variant}_rescued_samples",
        "description": (
            (
                f"VOC samples where defended {args.metric} improves over the undefended baseline by at least {args.min_delta:.4f}."
                if attack_config is None
                else (
                    f"VOC samples where defended {args.metric} under attack `{attack_config.name}` improves over the "
                    f"undefended baseline by at least {args.min_delta:.4f}."
                )
            )
        ),
        "dataset_family": "pascal_voc",
        "dataset": {
            "root_hint": args.dataset_root,
            "split": args.dataset_split,
            "base_dir": "VOCdevkit/VOC2012",
        },
        "model": {
            "family": args.family,
            "checkpoint": str(checkpoint_path.resolve()),
            "defense_config": str(defense_config_path.resolve()),
            "sparse_defense": sparse_info,
        },
        "attack": (
            None
            if attack_config is None
            else {
                "config": str(Path(args.attack_config).resolve()),
                "name": attack_config.name,
                "epsilon": attack_config.epsilon,
                "step_size": attack_config.resolved_step_size(),
                "steps": attack_config.steps,
                "random_start": attack_config.random_start,
                "targeted": attack_config.targeted,
            }
        ),
        "criterion": {
            "metric": args.metric,
            "min_delta": args.min_delta,
            "definition": "defended_metric - baseline_metric >= min_delta",
        },
        "counts": {
            "baseline_samples": len(baseline_rows),
            "defended_samples": len(defended_rows),
            "paired_samples": min(len(baseline_rows), len(defended_rows)),
            "rescued_samples": len(rescued_rows),
            "improved_samples": len(improved_rows),
        },
        "sample_ids": [str(row["sample_id"]) for row in rescued_rows],
        "rescued_sample_ids": [str(row["sample_id"]) for row in rescued_rows],
        "improved_sample_ids": [str(row["sample_id"]) for row in improved_rows],
        "rescued_rows": rescued_rows,
        "top_improved_rows": improved_rows[:50],
    }
    write_json(output_json, payload)
    logger.info(
        "Finished rescued sample search: paired=%d rescued=%d improved=%d output=%s",
        payload["counts"]["paired_samples"],
        payload["counts"]["rescued_samples"],
        payload["counts"]["improved_samples"],
        output_json.resolve(),
    )
    print(output_json.resolve())
    print(f"rescued_samples={len(rescued_rows)} improved_samples={len(improved_rows)}")


if __name__ == "__main__":
    main()
