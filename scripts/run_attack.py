from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401

from src.attacks import ATTACK_BACKWARD_MODE_CHOICES, AttackConfig, AttackRunner
from src.common import setup_logger
from src.common.config import load_yaml
from src.datasets import PASCAL_VOC_CLASS_NAMES, PascalVOCValidationDataset
from src.evaluation import evaluate_adversarial_segmentation_model
from src.models import MODEL_FAMILY_CHOICES, TorchSegmentationModelAdapter, build_model_from_checkpoint
from src.reporting.exporter import write_csv, write_json, write_markdown
from src.robustness.visualization import save_layerwise_feature_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint under adversarial attack.")
    parser.add_argument("--attack-config", required=True, help="Path to the attack YAML config.")
    parser.add_argument("--family", required=True, choices=MODEL_FAMILY_CHOICES, help="Model family to instantiate.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--defense-config", default="", help="Optional sparse defense YAML config.")
    parser.add_argument("--dataset-root", default="datasets", help="VOC dataset root that contains VOCdevkit/.")
    parser.add_argument("--dataset-split", default="val", help="VOC split file name under ImageSets/Segmentation.")
    parser.add_argument("--output-dir", default="", help="Directory for outputs. Defaults to results/reports/voc_adv_eval/<checkpoint>_<attack>.")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument("--num-classes", type=int, default=21, help="Segmentation class count.")
    parser.add_argument("--max-batches", type=int, default=-1, help="Optional early-stop for debugging.")
    parser.add_argument(
        "--epsilon-scale",
        type=float,
        default=1.0,
        help="Scale epsilon and explicit step_size by this multiplier to slightly enlarge the perturbation budget.",
    )
    parser.add_argument(
        "--epsilon-radius-255",
        type=float,
        default=None,
        help="Override the Linf budget with an absolute radius specified in pixel-space units out of 255.",
    )
    parser.add_argument(
        "--epsilon-radius-255-sweep",
        type=float,
        nargs="+",
        default=None,
        help="Run a budget sweep over multiple absolute Linf radii in 255-space, e.g. --epsilon-radius-255-sweep 2 4 8.",
    )
    parser.add_argument(
        "--attack-backward-mode",
        choices=ATTACK_BACKWARD_MODE_CHOICES,
        default="default",
        help="Backward pass mode for sparse modules during white-box attacks.",
    )
    parser.add_argument(
        "--num-restarts",
        type=int,
        default=1,
        help="Number of attack restarts. Worst-case adversarial examples are selected sample-wise across restarts.",
    )
    parser.add_argument(
        "--eot-iters",
        type=int,
        default=1,
        help="Number of forward/backward samples used for EOT-style gradient averaging.",
    )
    parser.add_argument(
        "--feature-vis-samples",
        type=int,
        default=0,
        help="Export layer-wise clean/adversarial feature visualizations for the first N validation samples.",
    )
    parser.add_argument(
        "--feature-vis-layers",
        type=int,
        default=-1,
        help="Maximum number of feature layers to export per sample. Use -1 to export all shared layers.",
    )
    parser.add_argument(
        "--feature-vis-dir",
        default="",
        help="Directory for feature visualization outputs. Defaults to <output-dir>/feature_visualizations.",
    )
    parser.add_argument("--strict", dest="strict", action="store_true", help="Require exact checkpoint key match.")
    parser.add_argument("--no-strict", dest="strict", action="store_false", help="Allow missing or unexpected checkpoint keys.")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace, attack_name: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    checkpoint_stem = Path(args.checkpoint).stem
    suffix = "_budget_sweep" if args.epsilon_radius_255_sweep else ""
    return Path("results/reports/voc_adv_eval") / f"{checkpoint_stem}_{attack_name}{suffix}"


def resolve_feature_vis_dir(args: argparse.Namespace, output_dir: Path, *, run_label: str | None = None) -> Path:
    if args.feature_vis_dir:
        base_dir = Path(args.feature_vis_dir)
        if run_label is None:
            return base_dir
        return base_dir / run_label
    return output_dir / "feature_visualizations"


def format_radius_label(radius_255: float) -> str:
    return f"eps_{radius_255:g}_255".replace(".", "p")


def build_attack_run_specs(base_config: AttackConfig, args: argparse.Namespace) -> list[tuple[str | None, AttackConfig]]:
    if args.epsilon_radius_255 is not None and args.epsilon_radius_255_sweep:
        raise ValueError("Specify either --epsilon-radius-255 or --epsilon-radius-255-sweep, not both.")

    common_overrides = {
        "epsilon_scale": args.epsilon_scale,
        "attack_backward_mode": args.attack_backward_mode,
        "num_restarts": args.num_restarts,
        "eot_iters": args.eot_iters,
    }
    if args.epsilon_radius_255_sweep:
        return [
            (
                format_radius_label(radius_255),
                base_config.with_runtime_overrides(
                    epsilon_radius_255=radius_255,
                    **common_overrides,
                ),
            )
            for radius_255 in args.epsilon_radius_255_sweep
        ]
    return [
        (
            None,
            base_config.with_runtime_overrides(
                epsilon_radius_255=args.epsilon_radius_255,
                **common_overrides,
            ),
        )
    ]


def export_feature_visualizations(
    model: TorchSegmentationModelAdapter,
    attack_config: AttackConfig,
    dataset: PascalVOCValidationDataset,
    output_dir: Path,
    sample_limit: int,
    max_layers: int,
    logger,
) -> list[dict[str, object]]:
    attack_runner = AttackRunner(model)
    exported: list[dict[str, object]] = []
    total_samples = min(max(sample_limit, 0), len(dataset))
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_index in range(total_samples):
        image, target, filename = dataset[sample_index]
        images = image.unsqueeze(0).to(model.device)
        targets = target.unsqueeze(0).to(model.device)

        with torch.no_grad():
            _, clean_features = model.forward_with_features(images)
        attack_output = attack_runner.run(config=attack_config, images=images, targets=targets)
        with torch.no_grad():
            _, adversarial_features = model.forward_with_features(attack_output.adversarial_images)

        sample_key = Path(filename).stem or f"sample_{sample_index:04d}"
        metadata = save_layerwise_feature_visualizations(
            output_dir=output_dir,
            sample_key=sample_key,
            clean_image=images,
            adversarial_image=attack_output.adversarial_images,
            perturbation=attack_output.perturbation,
            clean_features=clean_features,
            adversarial_features=adversarial_features,
            max_layers=max_layers,
        )
        exported.append(metadata)
        logger.info(
            "Feature visualization exported: sample=%s layers=%d output_dir=%s",
            sample_key,
            len(metadata["layers"]),
            metadata["sample_dir"],
        )

    return exported


def write_single_run_outputs(
    *,
    args: argparse.Namespace,
    attack_config: AttackConfig,
    checkpoint_path: Path,
    output_dir: Path,
    logger,
    dataset: PascalVOCValidationDataset,
    dataloader: DataLoader,
    adapter: TorchSegmentationModelAdapter,
    sparse_defense_info,
    missing_keys: list[str],
    unexpected_keys: list[str],
    run_label: str | None,
) -> dict[str, object]:
    logger.info(
        (
            "Evaluating attack run: label=%s epsilon=%s step_size=%s steps=%s "
            "epsilon_radius_255=%s backward_mode=%s restarts=%s eot_iters=%s output_dir=%s"
        ),
        run_label or "<single-run>",
        attack_config.epsilon,
        attack_config.resolved_step_size(),
        attack_config.steps,
        attack_config.epsilon_radius_255(),
        attack_config.attack_backward_mode,
        attack_config.num_restarts,
        attack_config.eot_iters,
        output_dir.resolve(),
    )

    summary = evaluate_adversarial_segmentation_model(
        model=adapter,
        attack_config=attack_config,
        dataloader=dataloader,
        ignore_index=None,
        class_names=PASCAL_VOC_CLASS_NAMES,
        max_batches=args.max_batches,
        logger=logger,
    )
    feature_visualizations: list[dict[str, object]] = []
    if args.feature_vis_samples > 0:
        feature_vis_dir = resolve_feature_vis_dir(args, output_dir, run_label=run_label)
        feature_visualizations = export_feature_visualizations(
            model=adapter,
            attack_config=attack_config,
            dataset=dataset,
            output_dir=feature_vis_dir,
            sample_limit=args.feature_vis_samples,
            max_layers=args.feature_vis_layers,
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
            "split": args.dataset_split,
            "resize_short": 473,
            "crop_size": 473,
            "num_samples": len(dataset),
        },
        "artifacts": {
            "feature_visualizations": feature_visualizations,
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
            (
                f"- defense_config: {Path(args.defense_config).resolve()}"
                if args.defense_config
                else "- defense_config: <none>"
            ),
            f"- dataset_root: {Path(args.dataset_root).resolve()}",
            f"- dataset_split: {args.dataset_split}",
            f"- attack: {attack_config.name}",
            f"- epsilon: {attack_config.epsilon}",
            f"- epsilon_radius_255: {attack_config.epsilon_radius_255() if attack_config.epsilon_radius_255() is not None else '<none>'}",
            f"- effective_epsilon_scale: {payload['attack']['effective_epsilon_scale']}",
            f"- step_size: {attack_config.resolved_step_size()}",
            f"- steps: {attack_config.steps}",
            f"- attack_backward_mode: {payload['attack']['attack_backward_mode']}",
            f"- num_restarts: {payload['attack']['num_restarts']}",
            f"- eot_iters: {payload['attack']['eot_iters']}",
            f"- sample_wise_worst_case_over_restarts: {payload['attack']['sample_wise_worst_case_over_restarts']}",
            f"- restart_selection: {payload['attack']['restart_selection'] or '<single-run>'}",
            f"- selected_restart_histogram: {payload['attack']['selected_restart_histogram']}",
            f"- restart_mean_score_by_restart: {payload['attack']['restart_mean_score_by_restart']}",
            f"- random_start: {attack_config.random_start}",
            f"- targeted: {attack_config.targeted}",
            f"- processed_samples: {payload['processed_samples']}",
            f"- processed_batches: {payload['processed_batches']}",
            f"- feature_visualization_samples: {len(feature_visualizations)}",
            (
                f"- feature_visualization_dir: {resolve_feature_vis_dir(args, output_dir, run_label=run_label).resolve()}"
                if feature_visualizations
                else "- feature_visualization_dir: <disabled>"
            ),
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
        "Finished attack run: label=%s mIoU=%.2f mAcc=%.2f aAcc=%.2f output_dir=%s",
        run_label or "<single-run>",
        payload["reference_percent"]["mIoU"],
        payload["reference_percent"]["mAcc"],
        payload["reference_percent"]["aAcc"],
        output_dir.resolve(),
    )
    return payload


def write_budget_sweep_outputs(output_dir: Path, payload: dict[str, object]) -> None:
    runs = payload["runs"]
    write_json(output_dir / "budget_sweep_summary.json", payload)
    write_csv(output_dir / "budget_sweep_metrics.csv", runs)
    report_lines = [
        f"- attack: {payload['attack_name']}",
        f"- checkpoint: {payload['checkpoint']}",
        f"- num_runs: {len(runs)}",
        f"- backward_mode: {payload['attack_backward_mode']}",
        f"- num_restarts: {payload['num_restarts']}",
        f"- eot_iters: {payload['eot_iters']}",
        "",
        "## Budgets",
        *[
            (
                f"- {run['budget_label']}: eps255={run['epsilon_radius_255']}, "
                f"epsilon={run['epsilon']:.6f}, step_size={run['step_size']:.6f}, "
                f"mIoU={run['mIoU']:.2f}, mAcc={run['mAcc']:.2f}, aAcc={run['aAcc']:.2f}, "
                f"selected_restart_histogram={run['selected_restart_histogram']}, "
                f"output_dir={run['output_dir']}"
            )
            for run in runs
        ],
    ]
    write_markdown(output_dir / "budget_sweep_report.md", "VOC Adversarial Budget Sweep", report_lines)


def main() -> None:
    args = parse_args()
    base_attack_config = AttackConfig.from_dict(load_yaml(args.attack_config))
    attack_run_specs = build_attack_run_specs(base_attack_config, args)
    checkpoint_path = Path(args.checkpoint)
    output_dir = resolve_output_dir(args, base_attack_config.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"voc_adv_eval.{checkpoint_path.stem}.{base_attack_config.name}", output_dir / "evaluate.log")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if (args.epsilon_radius_255 is not None or args.epsilon_radius_255_sweep) and args.epsilon_scale != 1.0:
        logger.warning(
            "Absolute epsilon radius override (%s) overrides --epsilon-scale=%s; using the absolute radius override.",
            args.epsilon_radius_255 if args.epsilon_radius_255 is not None else args.epsilon_radius_255_sweep,
            args.epsilon_scale,
        )
    logger.info("Starting adversarial evaluation")
    logger.info(
        "family=%s checkpoint=%s device=%s attack=%s run_count=%s defense_config=%s",
        args.family,
        checkpoint_path.resolve(),
        device,
        base_attack_config.name,
        len(attack_run_specs),
        Path(args.defense_config).resolve() if args.defense_config else "<none>",
    )

    dataset = PascalVOCValidationDataset(args.dataset_root, split=args.dataset_split, resize_short=473, crop_size=473)
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
    adapter = TorchSegmentationModelAdapter(model=model, num_classes=args.num_classes, device=device)
    logger.info("Checkpoint loaded: missing_keys=%d unexpected_keys=%d", len(missing_keys), len(unexpected_keys))
    sparse_defense_info = getattr(model, "_sparse_defense_info", None)

    run_payloads: list[tuple[str | None, Path, AttackConfig, dict[str, object]]] = []
    for run_label, attack_config in attack_run_specs:
        run_output_dir = output_dir if run_label is None else output_dir / run_label
        run_output_dir.mkdir(parents=True, exist_ok=True)
        payload = write_single_run_outputs(
            args=args,
            attack_config=attack_config,
            checkpoint_path=checkpoint_path,
            output_dir=run_output_dir,
            logger=logger,
            dataset=dataset,
            dataloader=dataloader,
            adapter=adapter,
            sparse_defense_info=sparse_defense_info,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            run_label=run_label,
        )
        run_payloads.append((run_label, run_output_dir, attack_config, payload))
        print(
            f"{checkpoint_path.name}: attack={attack_config.name} "
            f"eps255={attack_config.epsilon_radius_255() if attack_config.epsilon_radius_255() is not None else '<yaml>'} "
            f"mIoU={payload['reference_percent']['mIoU']:.2f} "
            f"mAcc={payload['reference_percent']['mAcc']:.2f} "
            f"aAcc={payload['reference_percent']['aAcc']:.2f}"
        )

    if len(run_payloads) > 1:
        sweep_payload = {
            "attack_name": base_attack_config.name,
            "checkpoint": str(checkpoint_path.resolve()),
            "attack_backward_mode": run_payloads[0][2].attack_backward_mode,
            "num_restarts": run_payloads[0][2].num_restarts,
            "eot_iters": run_payloads[0][2].eot_iters,
            "runs": [
                {
                    "budget_label": run_label,
                    "epsilon_radius_255": attack_config.epsilon_radius_255(),
                    "epsilon": attack_config.epsilon,
                    "step_size": attack_config.resolved_step_size(),
                    "steps": attack_config.steps,
                    "mIoU": payload["reference_percent"]["mIoU"],
                    "mAcc": payload["reference_percent"]["mAcc"],
                    "aAcc": payload["reference_percent"]["aAcc"],
                    "mean_linf": payload["attack"]["mean_linf"],
                    "max_linf": payload["attack"]["max_linf"],
                    "mean_l2": payload["attack"]["mean_l2"],
                    "best_mean_score": payload["attack"]["best_mean_score"],
                    "selected_restart_histogram": payload["attack"]["selected_restart_histogram"],
                    "selected_restart_fraction": payload["attack"]["selected_restart_fraction"],
                    "restart_mean_score_by_restart": payload["attack"]["restart_mean_score_by_restart"],
                    "runtime_samples_aggregated": payload["attack"]["runtime_samples_aggregated"],
                    "output_dir": str(run_output_dir.resolve()),
                }
                for run_label, run_output_dir, attack_config, payload in run_payloads
            ],
        }
        write_budget_sweep_outputs(output_dir, sweep_payload)
        print(f"Budget sweep outputs written to: {output_dir}")
    else:
        print(f"Outputs written to: {run_payloads[0][1]}")


if __name__ == "__main__":
    main()
