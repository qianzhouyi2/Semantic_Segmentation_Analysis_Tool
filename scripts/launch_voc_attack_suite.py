from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit the 18-model VOC attack suite jobs.")
    parser.add_argument("--manifest", required=True, help="attack_suite_manifest.json path.")
    parser.add_argument("--suite-root", required=True, help="Output root for the attack suite.")
    parser.add_argument("--dataset-root", default="datasets", help="VOC dataset root.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def submit(command: list[str]) -> int:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    if stdout:
        print(stdout)
    return int(stdout.split()[-1])


def submit_eval_case(
    *,
    mode: str,
    model: dict,
    output_dir: Path,
    dataset_root: str,
    attack_config: str | None = None,
    batch_size: int = 4,
) -> int:
    export_items = [
        "ALL",
        f"MODE={mode}",
        f"MODEL_ID={model['model_id']}",
        f"FAMILY={model['family']}",
        f"CHECKPOINT={model['checkpoint']}",
        f"OUTPUT_DIR={output_dir}",
        f"DATASET_ROOT={dataset_root}",
        f"NUM_WORKERS=4",
        f"BATCH_SIZE={batch_size}",
        "DEVICE=cuda",
    ]
    if model["defense_config"]:
        export_items.append(f"DEFENSE_CONFIG={model['defense_config']}")
    if attack_config is not None:
        export_items.append(f"ATTACK_CONFIG={attack_config}")
    return submit(
        [
            "sbatch",
            "--export=" + ",".join(export_items),
            "scripts/submit_voc_eval_case.sbatch",
        ]
    )


def submit_transfer_case(
    *,
    attack_config: str,
    source_model: dict,
    target_model: dict,
    output_dir: Path,
    dataset_root: str,
) -> int:
    export_items = [
        "ALL",
        f"ATTACK_CONFIG={attack_config}",
        f"SOURCE_MODEL_ID={source_model['model_id']}",
        f"SOURCE_FAMILY={source_model['family']}",
        f"SOURCE_CHECKPOINT={source_model['checkpoint']}",
        f"TARGET_MODEL_ID={target_model['model_id']}",
        f"TARGET_FAMILY={target_model['family']}",
        f"TARGET_CHECKPOINT={target_model['checkpoint']}",
        f"OUTPUT_DIR={output_dir}",
        f"DATASET_ROOT={dataset_root}",
        "NUM_WORKERS=4",
        "BATCH_SIZE=1",
        "DEVICE=cuda",
    ]
    if source_model["defense_config"]:
        export_items.append(f"SOURCE_DEFENSE_CONFIG={source_model['defense_config']}")
    if target_model["defense_config"]:
        export_items.append(f"TARGET_DEFENSE_CONFIG={target_model['defense_config']}")
    return submit(
        [
            "sbatch",
            "--export=" + ",".join(export_items),
            "scripts/submit_voc_transfer_attack_case.sbatch",
        ]
    )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    suite_root = Path(args.suite_root)
    suite_root.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path)
    models = manifest["models"]
    model_index = {item["model_id"]: item for item in models}

    job_ids: list[int] = []

    for model in models:
        if model["variant"] == "baseline":
            clean_output = suite_root / "clean" / model["model_id"]
            job_ids.append(
                submit_eval_case(
                    mode="clean",
                    model=model,
                    output_dir=clean_output,
                    dataset_root=args.dataset_root,
                    batch_size=8,
                )
            )
            pgd_output = suite_root / "whitebox" / "pgd" / model["model_id"]
            job_ids.append(
                submit_eval_case(
                    mode="attack",
                    model=model,
                    output_dir=pgd_output,
                    dataset_root=args.dataset_root,
                    attack_config="configs/attacks/pgd.yaml",
                    batch_size=2,
                )
            )

        segpgd_output = suite_root / "whitebox" / "segpgd" / model["model_id"]
        job_ids.append(
            submit_eval_case(
                mode="attack",
                model=model,
                output_dir=segpgd_output,
                dataset_root=args.dataset_root,
                attack_config="configs/attacks/segpgd.yaml",
                batch_size=2,
            )
        )

        source_model = model_index[model["transfer_source_model_id"]]
        mi_output = suite_root / "blackbox_transfer" / "mi_fgsm" / model["model_id"]
        niditi_output = suite_root / "blackbox_transfer" / "ni_di_ti" / model["model_id"]
        job_ids.append(
            submit_transfer_case(
                attack_config="configs/attacks/mi_fgsm.yaml",
                source_model=source_model,
                target_model=model,
                output_dir=mi_output,
                dataset_root=args.dataset_root,
            )
        )
        job_ids.append(
            submit_transfer_case(
                attack_config="configs/attacks/ni_di_ti.yaml",
                source_model=source_model,
                target_model=model,
                output_dir=niditi_output,
                dataset_root=args.dataset_root,
            )
        )

    dependency = ":".join(str(job_id) for job_id in job_ids)
    summary_job_id = submit(
        [
            "sbatch",
            f"--dependency=afterok:{dependency}",
            "--export=ALL,"
            + f"MANIFEST={manifest_path.resolve()},SUITE_ROOT={suite_root.resolve()}",
            "scripts/submit_voc_attack_suite_summary.sbatch",
        ]
    )
    print(
        json.dumps(
            {
                "num_jobs": len(job_ids),
                "job_ids": job_ids,
                "summary_job_id": summary_job_id,
                "suite_root": str(suite_root.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
