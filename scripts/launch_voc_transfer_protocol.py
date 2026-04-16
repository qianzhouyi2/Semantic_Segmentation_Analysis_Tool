from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit strict transfer-protocol VOC experiment jobs.")
    parser.add_argument("--manifest", required=True, help="transfer protocol manifest path.")
    parser.add_argument("--suite-root", required=True, help="Output root for transfer protocol.")
    parser.add_argument("--dataset-root", default="datasets", help="VOC dataset root.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_case_jsons(manifest: dict, case_dir: Path) -> list[Path]:
    case_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for case in manifest["cases"]:
        path = case_dir / f"{case['case_id']}.json"
        path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def submit(command: list[str]) -> int:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    if stdout:
        print(stdout)
    return int(stdout.split()[-1])


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    suite_root = Path(args.suite_root)
    suite_root.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path)
    case_json_dir = suite_root / "case_json"
    case_json_paths = write_case_jsons(manifest, case_json_dir)

    job_ids: list[int] = []
    for case_json_path, case in zip(case_json_paths, manifest["cases"], strict=True):
        output_dir = suite_root / "cases" / case["attack_stem"] / case["source"]["model_id"] / case["targets"][0]["display_name"]
        job_ids.append(
            submit(
                [
                    "sbatch",
                    "--export=ALL,"
                    + f"CASE_JSON={case_json_path.resolve()},OUTPUT_DIR={output_dir.resolve()},DATASET_ROOT={args.dataset_root},BATCH_SIZE=1,NUM_WORKERS=4,DEVICE=cuda",
                    "scripts/submit_voc_transfer_protocol_case.sbatch",
                ]
            )
        )

    dependency = ":".join(str(job_id) for job_id in job_ids)
    summary_job_id = submit(
        [
            "sbatch",
            f"--dependency=afterok:{dependency}",
            "--export=ALL,"
            + f"MANIFEST={manifest_path.resolve()},SUITE_ROOT={suite_root.resolve()}",
            "scripts/submit_voc_transfer_protocol_summary.sbatch",
        ]
    )
    print(
        json.dumps(
            {
                "num_cases": len(job_ids),
                "job_ids": job_ids,
                "summary_job_id": summary_job_id,
                "suite_root": str(suite_root.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
