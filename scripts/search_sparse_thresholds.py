from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import DataLoader

from src.attacks import AttackConfig
from src.common import setup_logger
from src.common.config import load_yaml
from src.common.sparse_workflow import (
    extract_variant_hyperparameters,
    resolve_sparse_defense_config,
    serialize_sparse_defense_config,
)
from src.datasets import PASCAL_VOC_CLASS_NAMES, PascalVOCValidationDataset
from src.evaluation import evaluate_adversarial_segmentation_model, evaluate_segmentation_model
from src.models import (
    MODEL_FAMILY_CHOICES,
    SPARSE_DEFENSE_CHOICES,
    TorchSegmentationModelAdapter,
    build_model_from_checkpoint,
)
from src.reporting.exporter import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search sparse-defense thresholds on a VOC split.")
    parser.add_argument("--family", required=True, choices=MODEL_FAMILY_CHOICES, help="Model family.")
    parser.add_argument("--checkpoint", required=True, help="Base checkpoint path.")
    parser.add_argument("--variant", required=True, choices=SPARSE_DEFENSE_CHOICES, help="Sparse variant.")
    parser.add_argument("--stats-path", required=True, help="Prepared sparse sidecar path.")
    parser.add_argument(
        "--defense-template-config",
        default="",
        help="Optional sparse defense YAML template. Search overrides threshold and stats_path while inheriting the other parameters.",
    )
    parser.add_argument("--attack-config", default="configs/attacks/pgd.yaml", help="Attack config YAML for search.")
    parser.add_argument("--dataset-root", default="datasets", help="VOC dataset root.")
    parser.add_argument("--dataset-split", default="train", help="VOC split file under ImageSets/Segmentation.")
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.35,0.40", help="Comma or space separated thresholds.")
    parser.add_argument("--output-dir", required=True, help="Output directory for this model/variant search.")
    parser.add_argument("--batch-size-clean", type=int, default=8, help="Batch size for clean eval.")
    parser.add_argument("--batch-size-adv", type=int, default=2, help="Batch size for adversarial eval.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--num-classes", type=int, default=21, help="Number of classes.")
    parser.add_argument("--max-batches", type=int, default=-1, help="Optional early stop for debugging.")
    parser.add_argument("--skip-clean", action="store_true", help="Only run PGD search without clean evaluation.")
    parser.add_argument("--strict", dest="strict", action="store_true", help="Strict checkpoint loading.")
    parser.add_argument("--no-strict", dest="strict", action="store_false", help="Allow checkpoint mismatch.")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    values = [item.strip() for chunk in raw.split(",") for item in chunk.split()]
    thresholds = [float(item) for item in values if item]
    if not thresholds:
        raise ValueError("No thresholds provided.")
    return thresholds


def slug_for_threshold(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "_")


def load_attack_config(path: str | Path) -> AttackConfig:
    return AttackConfig.from_dict(load_yaml(path))


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


def build_effective_defense_config(
    *,
    family: str,
    variant: str,
    threshold: float,
    stats_path: Path,
    defense_template_config: Path | None,
) -> tuple[dict, dict]:
    config = resolve_sparse_defense_config(
        variant=variant,
        family=family,
        threshold=threshold,
        stats_path=stats_path,
        template_config_path=defense_template_config,
    )
    return (
        serialize_sparse_defense_config(config, include_variant_alias=True),
        extract_variant_hyperparameters(config),
    )


def evaluate_threshold(
    *,
    family: str,
    checkpoint_path: Path,
    defense_config: dict,
    variant_hyperparameters: dict,
    attack_config: AttackConfig,
    dataset_root: str,
    dataset_split: str,
    batch_size_clean: int,
    batch_size_adv: int,
    num_workers: int,
    device: torch.device,
    num_classes: int,
    strict: bool,
    max_batches: int,
    logger,
) -> tuple[dict, dict]:
    model, missing_keys, unexpected_keys = build_model_from_checkpoint(
        family=family,
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        map_location="cpu",
        strict=strict,
        defense_config=defense_config,
    )
    sparse_info = getattr(model, "_sparse_defense_info", None)

    _, clean_loader = build_dataloader(
        dataset_root=dataset_root,
        dataset_split=dataset_split,
        batch_size=batch_size_clean,
        num_workers=num_workers,
        device=device,
    )
    clean_summary = evaluate_segmentation_model(
        model=model,
        dataloader=clean_loader,
        num_classes=num_classes,
        device=device,
        ignore_index=255,
        class_names=PASCAL_VOC_CLASS_NAMES,
        max_batches=max_batches,
        logger=logger,
    )

    _, adv_loader = build_dataloader(
        dataset_root=dataset_root,
        dataset_split=dataset_split,
        batch_size=batch_size_adv,
        num_workers=num_workers,
        device=device,
    )
    adapter = TorchSegmentationModelAdapter(model=model, num_classes=num_classes, device=device)
    adv_summary = evaluate_adversarial_segmentation_model(
        model=adapter,
        attack_config=attack_config,
        dataloader=adv_loader,
        ignore_index=255,
        class_names=PASCAL_VOC_CLASS_NAMES,
        max_batches=max_batches,
        logger=logger,
    )

    metadata = {
        "family": family,
        "checkpoint": str(checkpoint_path.resolve()),
        "variant": str(defense_config["variant"]),
        "stats_path": str(defense_config["stats_path"]),
        "threshold": float(defense_config["threshold"]),
        "defense_config": defense_config,
        "variant_hyperparameters": variant_hyperparameters,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "sparse_defense": sparse_info,
    }
    return (
        {"model": metadata, "dataset": {"root": str(Path(dataset_root).resolve()), "split": dataset_split}, **clean_summary},
        {"model": metadata, "dataset": {"root": str(Path(dataset_root).resolve()), "split": dataset_split}, **adv_summary},
    )


def compute_pareto_frontier(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    frontier: list[dict] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                other["clean_miou"] >= row["clean_miou"]
                and other["adv_miou"] >= row["adv_miou"]
                and (other["clean_miou"] > row["clean_miou"] or other["adv_miou"] > row["adv_miou"])
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    frontier.sort(key=lambda item: (item["threshold"], item["clean_miou"], item["adv_miou"]))
    return frontier


def _normalize(value: float, value_min: float, value_max: float) -> float:
    if abs(value_max - value_min) < 1e-12:
        return 1.0
    return (value - value_min) / (value_max - value_min)


def choose_best(rows: list[dict]) -> tuple[dict, list[dict]]:
    if not rows:
        raise ValueError("No threshold rows to rank.")
    if all(row["clean_miou"] is None for row in rows):
        best = max(rows, key=lambda row: (row["adv_miou"], -row["threshold"]))
        return best, [best]

    frontier = compute_pareto_frontier(rows)
    clean_values = [row["clean_miou"] for row in rows]
    adv_values = [row["adv_miou"] for row in rows]
    clean_min, clean_max = min(clean_values), max(clean_values)
    adv_min, adv_max = min(adv_values), max(adv_values)

    ranked: list[tuple[float, dict]] = []
    for row in frontier:
        clean_norm = _normalize(float(row["clean_miou"]), clean_min, clean_max)
        adv_norm = _normalize(float(row["adv_miou"]), adv_min, adv_max)
        distance_to_ideal = math.sqrt((1.0 - clean_norm) ** 2 + (1.0 - adv_norm) ** 2)
        row["pareto_clean_norm"] = clean_norm
        row["pareto_adv_norm"] = adv_norm
        row["pareto_distance_to_ideal"] = distance_to_ideal
        ranked.append((distance_to_ideal, row))

    ranked.sort(key=lambda item: (item[0], -item[1]["adv_miou"], -item[1]["clean_miou"], item[1]["threshold"]))
    return ranked[0][1], frontier


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    checkpoint_path = Path(args.checkpoint)
    stats_path = Path(args.stats_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        f"threshold_search.{checkpoint_path.stem}.{args.variant}.{args.dataset_split}",
        output_dir / "search.log",
    )
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    attack_config = load_attack_config(args.attack_config)
    defense_template_config = None if not args.defense_template_config else Path(args.defense_template_config)

    logger.info(
        "Starting threshold search: family=%s checkpoint=%s variant=%s split=%s stats=%s template=%s thresholds=%s device=%s",
        args.family,
        checkpoint_path.resolve(),
        args.variant,
        args.dataset_split,
        stats_path.resolve(),
        None if defense_template_config is None else defense_template_config.resolve(),
        thresholds,
        device,
    )

    rows: list[dict] = []
    for threshold in thresholds:
        logger.info("Evaluating threshold=%.4f", threshold)
        effective_defense_config, variant_hyperparameters = build_effective_defense_config(
            family=args.family,
            variant=args.variant,
            threshold=threshold,
            stats_path=stats_path,
            defense_template_config=defense_template_config,
        )
        clean_payload = None
        adv_payload = None
        if args.skip_clean:
            model, missing_keys, unexpected_keys = build_model_from_checkpoint(
                family=args.family,
                checkpoint_path=checkpoint_path,
                num_classes=args.num_classes,
                map_location="cpu",
                strict=args.strict,
                defense_config=effective_defense_config,
            )
            sparse_info = getattr(model, "_sparse_defense_info", None)
            _, adv_loader = build_dataloader(
                dataset_root=args.dataset_root,
                dataset_split=args.dataset_split,
                batch_size=args.batch_size_adv,
                num_workers=args.num_workers,
                device=device,
            )
            adapter = TorchSegmentationModelAdapter(model=model, num_classes=args.num_classes, device=device)
            adv_payload = evaluate_adversarial_segmentation_model(
                model=adapter,
                attack_config=attack_config,
                dataloader=adv_loader,
                ignore_index=255,
                class_names=PASCAL_VOC_CLASS_NAMES,
                max_batches=args.max_batches,
                logger=logger,
            )
            adv_payload = {
                "model": {
                    "family": args.family,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "variant": args.variant,
                    "stats_path": str(stats_path.resolve()),
                    "threshold": threshold,
                    "defense_config": effective_defense_config,
                    "variant_hyperparameters": variant_hyperparameters,
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "sparse_defense": sparse_info,
                },
                "dataset": {"root": str(Path(args.dataset_root).resolve()), "split": args.dataset_split},
                **adv_payload,
            }
        else:
            clean_payload, adv_payload = evaluate_threshold(
                family=args.family,
                checkpoint_path=checkpoint_path,
                defense_config=effective_defense_config,
                variant_hyperparameters=variant_hyperparameters,
                attack_config=attack_config,
                dataset_root=args.dataset_root,
                dataset_split=args.dataset_split,
                batch_size_clean=args.batch_size_clean,
                batch_size_adv=args.batch_size_adv,
                num_workers=args.num_workers,
                device=device,
                num_classes=args.num_classes,
                strict=args.strict,
                max_batches=args.max_batches,
                logger=logger,
            )
        thr_slug = slug_for_threshold(threshold)
        adv_dir = output_dir / f"thr_{thr_slug}_pgd"
        if clean_payload is not None:
            clean_dir = output_dir / f"thr_{thr_slug}_clean"
            write_json(clean_dir / "results.json", clean_payload)
        write_json(adv_dir / "results.json", adv_payload)

        rows.append(
            {
                "threshold": threshold,
                "clean_miou": None if clean_payload is None else float(clean_payload["reference_percent"]["mIoU"]),
                "clean_macc": None if clean_payload is None else float(clean_payload["reference_percent"]["mAcc"]),
                "clean_aacc": None if clean_payload is None else float(clean_payload["reference_percent"]["aAcc"]),
                "adv_miou": float(adv_payload["reference_percent"]["mIoU"]),
                "adv_macc": float(adv_payload["reference_percent"]["mAcc"]),
                "adv_aacc": float(adv_payload["reference_percent"]["aAcc"]),
                "clean_results": None if clean_payload is None else str((clean_dir / "results.json").resolve()),
                "adv_results": str((adv_dir / "results.json").resolve()),
                "effective_defense_config": effective_defense_config,
                "variant_hyperparameters": variant_hyperparameters,
            }
        )

    best, pareto_frontier = choose_best(rows)
    summary = {
        "family": args.family,
        "checkpoint": str(checkpoint_path.resolve()),
        "variant": args.variant,
        "stats_path": str(stats_path.resolve()),
        "defense_template_config": (
            None if defense_template_config is None else str(defense_template_config.resolve())
        ),
        "effective_defense_config": best["effective_defense_config"],
        "variant_hyperparameters": best["variant_hyperparameters"],
        "dataset": {
            "root": str(Path(args.dataset_root).resolve()),
            "split": args.dataset_split,
        },
        "attack_config": str(Path(args.attack_config).resolve()),
        "skip_clean": bool(args.skip_clean),
        "thresholds": rows,
        "pareto_frontier": pareto_frontier,
        "best_threshold": best,
        "selection_rule": (
            "max adv_miou, then lower threshold"
            if args.skip_clean
            else "Pareto frontier on (clean_miou, adv_miou), then choose the point with minimum normalized distance to ideal (1,1); tie-break by higher adv_miou, higher clean_miou, lower threshold"
        ),
    }
    write_json(output_dir / "search_summary.json", summary)
    write_markdown(
        output_dir / "search_summary.md",
        f"Threshold Search {checkpoint_path.stem} {args.variant}",
        [
            f"- family: {args.family}",
            f"- checkpoint: {checkpoint_path.resolve()}",
            f"- variant: {args.variant}",
            f"- stats_path: {stats_path.resolve()}",
            (
                "- defense_template_config: -"
                if defense_template_config is None
                else f"- defense_template_config: {defense_template_config.resolve()}"
            ),
            f"- dataset_split: {args.dataset_split}",
            f"- attack_config: {Path(args.attack_config).resolve()}",
            f"- best_threshold: {best['threshold']:.2f}",
            f"- best_adv_mIoU: {best['adv_miou']:.2f}",
            (
                "- best_clean_mIoU: -"
                if args.skip_clean
                else f"- best_clean_mIoU: {best['clean_miou']:.2f}"
            ),
            f"- variant_hyperparameters: {json.dumps(best['variant_hyperparameters'], ensure_ascii=False)}",
            f"- pareto_frontier_size: {len(pareto_frontier)}",
            "",
            "## Thresholds",
            "",
            (
                "| threshold | adv mIoU | adv mAcc |"
                if args.skip_clean
                else "| threshold | clean mIoU | adv mIoU | clean mAcc | adv mAcc |"
            ),
            (
                "| --- | ---: | ---: |"
                if args.skip_clean
                else "| --- | ---: | ---: | ---: | ---: |"
            ),
            *(
                [
                    f"| {row['threshold']:.2f} | {row['adv_miou']:.2f} | {row['adv_macc']:.2f} |"
                    for row in rows
                ]
                if args.skip_clean
                else [
                    f"| {row['threshold']:.2f} | {row['clean_miou']:.2f} | {row['adv_miou']:.2f} | {row['clean_macc']:.2f} | {row['adv_macc']:.2f} |"
                    for row in rows
                ]
            ),
            "",
            "## Pareto Frontier",
            "",
            (
                "- clean search skipped"
                if args.skip_clean
                else "| threshold | clean mIoU | PGD mIoU | dist to ideal |"
            ),
            (
                ""
                if args.skip_clean
                else "| --- | ---: | ---: | ---: |"
            ),
            *(
                []
                if args.skip_clean
                else [
                    f"| {row['threshold']:.2f} | {row['clean_miou']:.2f} | {row['adv_miou']:.2f} | {row['pareto_distance_to_ideal']:.4f} |"
                    for row in pareto_frontier
                ]
            ),
        ],
    )
    print(json.dumps({"best_threshold": best["threshold"], "output_dir": str(output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
