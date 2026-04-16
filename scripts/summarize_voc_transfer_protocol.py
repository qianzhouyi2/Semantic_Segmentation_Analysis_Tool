from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.reporting.exporter import write_csv, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize strict transfer-protocol experiment results.")
    parser.add_argument("--manifest", required=True, help="transfer_protocol_manifest.json path.")
    parser.add_argument("--suite-root", required=True, help="Transfer suite output root.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    suite_root = Path(args.suite_root)
    manifest = load_json(manifest_path)

    case_rows: list[dict] = []
    target_rows: list[dict] = []
    for case in manifest["cases"]:
        case_dir = suite_root / "cases" / case["attack_stem"] / case["source"]["model_id"] / case["targets"][0]["display_name"]
        summary_path = case_dir / "summary.json"
        if not summary_path.exists():
            continue
        payload = load_json(summary_path)
        source_self = payload["source_self"]
        case_rows.append(
            {
                "case_id": payload["case_id"],
                "regime": payload["regime"],
                "relation": payload["relation"],
                "attack_stem": payload["attack"]["stem"],
                "source_model_id": payload["source_self"]["model"]["model_id"],
                "source_clean_miou": payload["source_self"]["clean"]["reference_percent"]["mIoU"],
                "source_transfer_miou": payload["source_self"]["transfer"]["reference_percent"]["mIoU"],
                "source_transfer_drop": payload["source_self"]["transfer_miou_drop"],
                "summary_path": str(summary_path.resolve()),
            }
        )
        for target in payload["targets"]:
            target_rows.append(
                {
                    "case_id": payload["case_id"],
                    "regime": payload["regime"],
                    "relation": payload["relation"],
                    "attack_stem": payload["attack"]["stem"],
                    "source_model_id": payload["source_self"]["model"]["model_id"],
                    "target_model_id": target["model"]["model_id"],
                    "target_family": target["model"]["family"],
                    "target_variant": target["model"]["sparse_defense"]["variant"] if target["model"]["sparse_defense"] else "baseline",
                    "clean_miou": target["clean"]["reference_percent"]["mIoU"],
                    "transfer_miou": target["transfer"]["reference_percent"]["mIoU"],
                    "transfer_miou_drop": target["transfer_miou_drop"],
                }
            )

    output_dir = suite_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": str(manifest_path.resolve()),
        "suite_root": str(suite_root.resolve()),
        "num_cases": len(case_rows),
        "case_rows": case_rows,
        "target_rows": target_rows,
    }
    write_json(output_dir / "transfer_protocol_summary.json", payload)
    write_csv(output_dir / "transfer_protocol_cases.csv", case_rows)
    write_csv(output_dir / "transfer_protocol_targets.csv", target_rows)
    write_markdown(
        output_dir / "transfer_protocol_summary.md",
        "VOC Transfer Protocol Summary",
        [
            f"- manifest: {manifest_path.resolve()}",
            f"- suite_root: {suite_root.resolve()}",
            f"- num_cases: {len(case_rows)}",
            f"- num_target_rows: {len(target_rows)}",
            "",
            "| case_id | regime | relation | attack | source | source clean | source transfer | source drop |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
            *[
                f"| `{row['case_id']}` | `{row['regime']}` | `{row['relation']}` | `{row['attack_stem']}` | "
                f"`{row['source_model_id']}` | {row['source_clean_miou']:.2f} | {row['source_transfer_miou']:.2f} | {row['source_transfer_drop']:.2f} |"
                for row in case_rows
            ],
        ],
    )
    print(output_dir / "transfer_protocol_summary.json")


if __name__ == "__main__":
    main()
