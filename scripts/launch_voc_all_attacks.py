from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit full all-attacks VOC evaluation jobs for 18 models.")
    parser.add_argument("--manifest", required=True, help="attack suite manifest path.")
    parser.add_argument("--suite-root", required=True, help="Output root for all-attacks suite.")
    parser.add_argument("--dataset-root", default="datasets", help="VOC dataset root.")
    parser.add_argument("--attack-config-dir", default="configs/attacks", help="Attack config directory.")
    parser.add_argument(
        "--exclude-attack-stems",
        default="",
        help="Comma or space separated attack config stems to exclude.",
    )
    parser.add_argument(
        "--save-per-sample",
        action="store_true",
        help="Export per-sample CSV/JSONL for each clean and attack run.",
    )
    parser.add_argument(
        "--per-sample-policy",
        choices=("auto", "require", "skip"),
        default="auto",
        help="Worst-case aggregation policy passed to summarize_voc_all_attacks.py.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def submit(command: list[str]) -> int:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    if stdout:
        print(stdout)
    return int(stdout.split()[-1])


def discover_attack_configs(config_dir: Path) -> list[Path]:
    return [path for path in sorted(config_dir.glob("*.yaml")) if path.stem != "eval"]


def parse_excluded_stems(raw: str) -> set[str]:
    return {item.strip() for chunk in raw.split(",") for item in chunk.split() if item.strip()}


def submit_eval_case(
    *,
    mode: str,
    model: dict,
    output_dir: Path,
    dataset_root: str,
    attack_config: Path | None = None,
    batch_size: int = 4,
    save_per_sample: bool = False,
) -> int:
    export_items = [
        "ALL",
        f"MODE={mode}",
        f"MODEL_ID={model['model_id']}",
        f"FAMILY={model['family']}",
        f"CHECKPOINT={model['checkpoint']}",
        f"OUTPUT_DIR={output_dir}",
        f"DATASET_ROOT={dataset_root}",
        "NUM_WORKERS=4",
        f"BATCH_SIZE={batch_size}",
        "DEVICE=cuda",
    ]
    if model["defense_config"]:
        export_items.append(f"DEFENSE_CONFIG={model['defense_config']}")
    if attack_config is not None:
        export_items.append(f"ATTACK_CONFIG={attack_config.resolve()}")
    if save_per_sample:
        export_items.append("SAVE_PER_SAMPLE=1")
    return submit(
        [
            "sbatch",
            "--export=" + ",".join(export_items),
            "scripts/submit_voc_eval_case.sbatch",
        ]
    )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    suite_root = Path(args.suite_root)
    suite_root.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path)
    models = manifest["models"]
    excluded_stems = parse_excluded_stems(args.exclude_attack_stems)
    attack_configs = [path for path in discover_attack_configs(Path(args.attack_config_dir)) if path.stem not in excluded_stems]

    job_ids: list[int] = []
    for model in models:
        clean_output = suite_root / "clean" / model["model_id"]
        job_ids.append(
            submit_eval_case(
                mode="clean",
                model=model,
                output_dir=clean_output,
                dataset_root=args.dataset_root,
                batch_size=8,
                save_per_sample=args.save_per_sample,
            )
        )
        for attack_config in attack_configs:
            output_dir = suite_root / "attacks" / attack_config.stem / model["model_id"]
            attack_name = attack_config.stem
            batch_size = 1 if attack_name in {"fspgd", "dag"} else 2
            job_ids.append(
                submit_eval_case(
                    mode="attack",
                    model=model,
                    output_dir=output_dir,
                    dataset_root=args.dataset_root,
                    attack_config=attack_config,
                    batch_size=batch_size,
                    save_per_sample=args.save_per_sample,
                )
            )

    dependency = ":".join(str(job_id) for job_id in job_ids)
    summary_job_id = submit(
        [
            "sbatch",
            f"--dependency=afterok:{dependency}",
            "--export=ALL,"
            + (
                f"MANIFEST={manifest_path.resolve()},SUITE_ROOT={suite_root.resolve()},"
                f"PER_SAMPLE_POLICY={args.per_sample_policy}"
            ),
            "scripts/submit_voc_all_attacks_summary.sbatch",
        ]
    )
    print(
        json.dumps(
            {
                "num_jobs": len(job_ids),
                "job_ids": job_ids,
                "summary_job_id": summary_job_id,
                "suite_root": str(suite_root.resolve()),
                "num_attacks": len(attack_configs),
                "excluded_attack_stems": sorted(excluded_stems),
                "save_per_sample": args.save_per_sample,
                "per_sample_policy": args.per_sample_policy,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
