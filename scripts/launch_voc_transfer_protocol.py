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
    parser.add_argument("--epsilon-scale", type=float, default=1.0, help="Optional multiplicative Linf budget scale.")
    parser.add_argument(
        "--epsilon-radius-255",
        type=float,
        default=None,
        help="Optional absolute Linf budget override in 255-space passed to transfer attack jobs.",
    )
    parser.add_argument(
        "--attack-backward-mode",
        default="default",
        choices=("default", "bpda_ste"),
        help="Backward mode forwarded to transfer attack jobs.",
    )
    parser.add_argument(
        "--num-restarts",
        type=int,
        default=1,
        help="Restart count forwarded to transfer attack jobs.",
    )
    parser.add_argument(
        "--eot-iters",
        type=int,
        default=1,
        help="EOT gradient averaging count forwarded to transfer attack jobs.",
    )
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


def append_attack_protocol_exports(
    export_items: list[str],
    *,
    epsilon_scale: float = 1.0,
    epsilon_radius_255: float | None = None,
    attack_backward_mode: str = "default",
    num_restarts: int = 1,
    eot_iters: int = 1,
) -> list[str]:
    export_items.append(f"EPSILON_SCALE={epsilon_scale}")
    if epsilon_radius_255 is not None:
        export_items.append(f"EPSILON_RADIUS_255={epsilon_radius_255}")
    export_items.append(f"ATTACK_BACKWARD_MODE={attack_backward_mode}")
    export_items.append(f"NUM_RESTARTS={num_restarts}")
    export_items.append(f"EOT_ITERS={eot_iters}")
    return export_items


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
                    + ",".join(
                        append_attack_protocol_exports(
                            [
                                f"CASE_JSON={case_json_path.resolve()}",
                                f"OUTPUT_DIR={output_dir.resolve()}",
                                f"DATASET_ROOT={args.dataset_root}",
                                "BATCH_SIZE=1",
                                "NUM_WORKERS=4",
                                "DEVICE=cuda",
                            ],
                            epsilon_scale=args.epsilon_scale,
                            epsilon_radius_255=args.epsilon_radius_255,
                            attack_backward_mode=args.attack_backward_mode,
                            num_restarts=args.num_restarts,
                            eot_iters=args.eot_iters,
                        )
                    ),
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
