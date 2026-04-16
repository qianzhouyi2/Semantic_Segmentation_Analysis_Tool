from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.reporting.exporter import write_csv, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the 18-model VOC attack suite.")
    parser.add_argument("--manifest", required=True, help="attack_suite_manifest.json path.")
    parser.add_argument("--suite-root", required=True, help="Root directory for attack suite outputs.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_load_summary(path: Path) -> dict | None:
    return load_json(path) if path.exists() else None


def ref_miou(summary: dict | None) -> float | None:
    if summary is None:
        return None
    return float(summary["reference_percent"]["mIoU"])


def format_num(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def format_threshold(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    suite_root = Path(args.suite_root)
    manifest = load_json(manifest_path)

    rows: list[dict] = []
    for model in manifest["models"]:
        model_id = model["model_id"]
        clean_summary_path = (
            Path(model["reused_clean_summary"])
            if model["reused_clean_summary"] is not None
            else suite_root / "clean" / model_id / "summary.json"
        )
        pgd_summary_path = (
            Path(model["reused_pgd_summary"])
            if model["reused_pgd_summary"] is not None
            else suite_root / "whitebox" / "pgd" / model_id / "summary.json"
        )
        segpgd_summary_path = suite_root / "whitebox" / "segpgd" / model_id / "summary.json"
        mi_transfer_summary_path = suite_root / "blackbox_transfer" / "mi_fgsm" / model_id / "summary.json"
        niditi_transfer_summary_path = suite_root / "blackbox_transfer" / "ni_di_ti" / model_id / "summary.json"

        clean_summary = maybe_load_summary(clean_summary_path)
        pgd_summary = maybe_load_summary(pgd_summary_path)
        segpgd_summary = maybe_load_summary(segpgd_summary_path)
        mi_transfer_summary = maybe_load_summary(mi_transfer_summary_path)
        niditi_transfer_summary = maybe_load_summary(niditi_transfer_summary_path)

        row = {
            "model_id": model_id,
            "display_name": model["display_name"],
            "family": model["family"],
            "variant": model["variant"],
            "threshold": model["threshold"],
            "transfer_source_model_id": model["transfer_source_model_id"],
            "clean_miou": ref_miou(clean_summary),
            "pgd_miou": ref_miou(pgd_summary),
            "segpgd_miou": ref_miou(segpgd_summary),
            "mi_fgsm_transfer_miou": ref_miou(mi_transfer_summary),
            "ni_di_ti_transfer_miou": ref_miou(niditi_transfer_summary),
            "clean_summary": str(clean_summary_path.resolve()),
            "pgd_summary": str(pgd_summary_path.resolve()),
            "segpgd_summary": str(segpgd_summary_path.resolve()),
            "mi_fgsm_transfer_summary": str(mi_transfer_summary_path.resolve()),
            "ni_di_ti_transfer_summary": str(niditi_transfer_summary_path.resolve()),
        }
        if row["clean_miou"] is not None and row["pgd_miou"] is not None:
            row["pgd_drop"] = float(row["clean_miou"] - row["pgd_miou"])
        else:
            row["pgd_drop"] = None
        rows.append(row)

    output_dir = suite_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": str(manifest_path.resolve()),
        "suite_root": str(suite_root.resolve()),
        "num_models": len(rows),
        "rows": rows,
    }
    write_json(output_dir / "attack_suite_summary.json", payload)
    write_csv(output_dir / "attack_suite_summary.csv", rows)
    write_markdown(
        output_dir / "attack_suite_summary.md",
        "VOC Attack Suite Summary",
        [
            f"- manifest: {manifest_path.resolve()}",
            f"- suite_root: {suite_root.resolve()}",
            f"- num_models: {len(rows)}",
            "",
            "| model_id | family | variant | threshold | clean mIoU | PGD mIoU | SegPGD mIoU | MI-FGSM transfer mIoU | NI+DI+TI transfer mIoU |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| `{row['model_id']}` | `{row['family']}` | `{row['variant']}` | "
                f"{format_threshold(row['threshold'])} | "
                f"{format_num(row['clean_miou'])} | {format_num(row['pgd_miou'])} | {format_num(row['segpgd_miou'])} | "
                f"{format_num(row['mi_fgsm_transfer_miou'])} | {format_num(row['ni_di_ti_transfer_miou'])} |"
                for row in rows
            ],
        ],
    )
    print(output_dir / "attack_suite_summary.json")


if __name__ == "__main__":
    main()
