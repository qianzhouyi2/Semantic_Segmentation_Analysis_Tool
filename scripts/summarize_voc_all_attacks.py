from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.reporting.exporter import write_csv, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize full VOC all-attacks evaluation results.")
    parser.add_argument("--manifest", required=True, help="attack suite manifest path.")
    parser.add_argument("--suite-root", required=True, help="Root directory for full attack suite.")
    parser.add_argument("--attack-config-dir", default="configs/attacks", help="Attack config directory.")
    parser.add_argument(
        "--exclude-attack-stems",
        default="",
        help="Comma or space separated attack config stems to exclude from the summary.",
    )
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


def discover_attack_stems(config_dir: Path) -> list[str]:
    stems = [path.stem for path in sorted(config_dir.glob("*.yaml")) if path.stem != "eval"]
    if not stems:
        raise FileNotFoundError(f"No attack configs found under {config_dir}")
    return stems


def parse_excluded_stems(raw: str) -> set[str]:
    return {item.strip() for chunk in raw.split(",") for item in chunk.split() if item.strip()}


def mean_attack_miou(row: dict, attack_stems: list[str]) -> float | None:
    values = [row.get(f"{stem}_miou") for stem in attack_stems]
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def build_aligned_markdown_table(rows: list[list[str]], *, right_align: set[int]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]

    def format_row(row: list[str]) -> str:
        cells = [row[index].ljust(widths[index]) for index in range(len(row))]
        return "| " + " | ".join(cells) + " |"

    def separator_cell(index: int) -> str:
        width = max(3, widths[index])
        if index in right_align:
            return "-" * (width - 1) + ":"
        return "-" * width

    separator = "| " + " | ".join(separator_cell(index) for index in range(len(widths))) + " |"
    return [format_row(rows[0]), separator, *(format_row(row) for row in rows[1:])]


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    suite_root = Path(args.suite_root)
    config_dir = Path(args.attack_config_dir)
    manifest = load_json(manifest_path)
    excluded_stems = parse_excluded_stems(args.exclude_attack_stems)
    attack_stems = [stem for stem in discover_attack_stems(config_dir) if stem not in excluded_stems]

    rows: list[dict] = []
    for model in manifest["models"]:
        model_id = model["model_id"]
        clean_summary = maybe_load_summary(suite_root / "clean" / model_id / "summary.json")
        row = {
            "model_id": model_id,
            "display_name": model["display_name"],
            "family": model["family"],
            "variant": model["variant"],
            "threshold": model["threshold"],
            "clean_miou": ref_miou(clean_summary),
            "clean_summary": None if clean_summary is None else str((suite_root / "clean" / model_id / "summary.json").resolve()),
        }
        for stem in attack_stems:
            attack_summary_path = suite_root / "attacks" / stem / model_id / "summary.json"
            attack_summary = maybe_load_summary(attack_summary_path)
            row[f"{stem}_miou"] = ref_miou(attack_summary)
            row[f"{stem}_summary"] = None if attack_summary is None else str(attack_summary_path.resolve())
        row["mean_attack_miou"] = mean_attack_miou(row, attack_stems)
        rows.append(row)

    output_dir = suite_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": str(manifest_path.resolve()),
        "suite_root": str(suite_root.resolve()),
        "attack_stems": attack_stems,
        "excluded_attack_stems": sorted(excluded_stems),
        "num_models": len(rows),
        "rows": rows,
    }
    write_json(output_dir / "all_attacks_summary.json", payload)
    write_csv(output_dir / "all_attacks_summary.csv", rows)

    table_rows = [[
        "model_id",
        "family",
        "variant",
        "threshold",
        "clean",
        *attack_stems,
        "mean",
    ]]
    for row in rows:
        values = [
            f"`{row['model_id']}`",
            f"`{row['family']}`",
            f"`{row['variant']}`",
            format_threshold(row["threshold"]),
            format_num(row["clean_miou"]),
        ]
        values.extend(format_num(row[f"{stem}_miou"]) for stem in attack_stems)
        values.append(format_num(row["mean_attack_miou"]))
        table_rows.append(values)

    table_lines = build_aligned_markdown_table(
        table_rows,
        right_align=set(range(3, len(table_rows[0]))),
    )

    write_markdown(
        output_dir / "all_attacks_summary.md",
        "VOC All Attacks Summary",
        [
            f"- manifest: {manifest_path.resolve()}",
            f"- suite_root: {suite_root.resolve()}",
            f"- num_models: {len(rows)}",
            f"- attacks: {', '.join(attack_stems)}",
            "",
            *table_lines,
        ],
    )
    print(output_dir / "all_attacks_summary.json")


if __name__ == "__main__":
    main()
