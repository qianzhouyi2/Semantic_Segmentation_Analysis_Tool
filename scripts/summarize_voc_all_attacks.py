from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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
    parser.add_argument(
        "--per-sample-policy",
        choices=("auto", "require", "skip"),
        default="auto",
        help=(
            "How to handle worst-case image-wise aggregation that depends on per-sample CSVs. "
            "`auto` computes worst-case only when every counted attack has aligned per-sample files. "
            "`require` raises when any counted attack is missing them. `skip` disables worst-case aggregation."
        ),
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


def resolve_per_sample_metrics_path(summary_path: Path, summary: dict | None) -> Path | None:
    if summary is None:
        return None
    artifacts = summary.get("artifacts")
    if isinstance(artifacts, dict):
        artifact_path = artifacts.get("per_sample_metrics_csv")
        if artifact_path:
            return Path(str(artifact_path))
    candidate = summary_path.parent / "per_sample_metrics.csv"
    return candidate if candidate.exists() else None


def load_per_sample_mious(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "sample_miou"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Per-sample CSV is missing required columns at {path}")
        values: dict[str, float] = {}
        for row in reader:
            filename = str(row["filename"])
            if filename in values:
                raise ValueError(f"Duplicate filename {filename!r} in {path}")
            values[filename] = float(row["sample_miou"])
    return values


def compute_imagewise_worstcase(
    attack_summaries: dict[str, tuple[Path, dict | None]],
    *,
    per_sample_policy: str,
) -> dict[str, object]:
    counted_attack_stems = [stem for stem, (_summary_path, summary) in attack_summaries.items() if summary is not None]
    result = {
        "worst_imagewise_attack_miou": None,
        "worst_attack_stem_by_frequency": None,
        "worst_attack_stem_histogram": None,
        "num_attacks_with_per_sample": 0,
        "missing_per_sample_attack_stems": [],
        "worstcase_status": "skipped" if per_sample_policy == "skip" else "not_available",
    }
    if per_sample_policy == "skip" or not counted_attack_stems:
        return result

    per_attack_mious: dict[str, dict[str, float]] = {}
    missing_stems: list[str] = []
    for stem in counted_attack_stems:
        summary_path, summary = attack_summaries[stem]
        per_sample_path = resolve_per_sample_metrics_path(summary_path, summary)
        if per_sample_path is None or not per_sample_path.exists():
            missing_stems.append(stem)
            continue
        per_attack_mious[stem] = load_per_sample_mious(per_sample_path)

    result["num_attacks_with_per_sample"] = len(per_attack_mious)
    result["missing_per_sample_attack_stems"] = missing_stems

    if missing_stems:
        if per_sample_policy == "require":
            raise FileNotFoundError(
                "Missing per-sample metrics for counted attacks: " + ", ".join(sorted(missing_stems))
            )
        result["worstcase_status"] = "missing_per_sample"
        return result

    filename_sets = {stem: set(values.keys()) for stem, values in per_attack_mious.items()}
    reference_stem = counted_attack_stems[0]
    reference_filenames = filename_sets[reference_stem]
    mismatched_stems = [stem for stem, filenames in filename_sets.items() if filenames != reference_filenames]
    if mismatched_stems:
        mismatch_list = ", ".join(sorted(mismatched_stems))
        raise ValueError(
            "Per-sample filename sets do not align across attacks. "
            f"Reference attack={reference_stem}, mismatched={mismatch_list}."
        )

    ordered_filenames = sorted(reference_filenames)
    if not ordered_filenames:
        result["worstcase_status"] = "empty"
        return result

    worst_scores: list[float] = []
    worst_attack_counts: Counter[str] = Counter()
    for filename in ordered_filenames:
        worst_stem, worst_score = min(
            ((stem, per_attack_mious[stem][filename]) for stem in counted_attack_stems),
            key=lambda item: (item[1], item[0]),
        )
        worst_scores.append(worst_score)
        worst_attack_counts[worst_stem] += 1

    most_frequent_worst_attack = sorted(
        worst_attack_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0][0]
    result["worst_imagewise_attack_miou"] = float(sum(worst_scores) / float(len(worst_scores)) * 100.0)
    result["worst_attack_stem_by_frequency"] = most_frequent_worst_attack
    result["worst_attack_stem_histogram"] = dict(sorted(worst_attack_counts.items()))
    result["worstcase_status"] = "computed"
    return result


def build_paper_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "model_id": row["model_id"],
            "display_name": row["display_name"],
            "family": row["family"],
            "variant": row["variant"],
            "threshold": row["threshold"],
            "clean_miou": row["clean_miou"],
            "mean_attack_miou": row["mean_attack_miou"],
            "worst_imagewise_attack_miou": row["worst_imagewise_attack_miou"],
            "worst_attack_stem_by_frequency": row["worst_attack_stem_by_frequency"],
            "num_attacks_counted": row["num_attacks_counted"],
            "num_attacks_with_per_sample": row["num_attacks_with_per_sample"],
            "worstcase_status": row["worstcase_status"],
        }
        for row in rows
    ]


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
        clean_summary_path = suite_root / "clean" / model_id / "summary.json"
        clean_summary = maybe_load_summary(clean_summary_path)
        row = {
            "model_id": model_id,
            "display_name": model["display_name"],
            "family": model["family"],
            "variant": model["variant"],
            "threshold": model["threshold"],
            "clean_miou": ref_miou(clean_summary),
            "clean_summary": None if clean_summary is None else str(clean_summary_path.resolve()),
        }
        attack_summary_by_stem: dict[str, tuple[Path, dict | None]] = {}
        for stem in attack_stems:
            attack_summary_path = suite_root / "attacks" / stem / model_id / "summary.json"
            attack_summary = maybe_load_summary(attack_summary_path)
            attack_summary_by_stem[stem] = (attack_summary_path, attack_summary)
            row[f"{stem}_miou"] = ref_miou(attack_summary)
            row[f"{stem}_summary"] = None if attack_summary is None else str(attack_summary_path.resolve())
        row["num_attacks_counted"] = sum(1 for stem in attack_stems if row[f"{stem}_miou"] is not None)
        row["mean_attack_miou"] = mean_attack_miou(row, attack_stems)
        worstcase = compute_imagewise_worstcase(
            attack_summary_by_stem,
            per_sample_policy=args.per_sample_policy,
        )
        row["worst_imagewise_attack_miou"] = worstcase["worst_imagewise_attack_miou"]
        row["worst_attack_stem_by_frequency"] = worstcase["worst_attack_stem_by_frequency"]
        row["worst_attack_stem_histogram"] = worstcase["worst_attack_stem_histogram"]
        row["num_attacks_with_per_sample"] = worstcase["num_attacks_with_per_sample"]
        row["missing_per_sample_attack_stems"] = ",".join(worstcase["missing_per_sample_attack_stems"])
        row["worstcase_status"] = worstcase["worstcase_status"]
        rows.append(row)

    output_dir = suite_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": str(manifest_path.resolve()),
        "suite_root": str(suite_root.resolve()),
        "attack_stems": attack_stems,
        "excluded_attack_stems": sorted(excluded_stems),
        "num_models": len(rows),
        "per_sample_policy": args.per_sample_policy,
        "rows_with_computed_worstcase": sum(1 for row in rows if row["worstcase_status"] == "computed"),
        "rows": rows,
    }
    paper_rows = build_paper_rows(rows)

    paper_payload = {
        "manifest": payload["manifest"],
        "suite_root": payload["suite_root"],
        "attack_stems": payload["attack_stems"],
        "excluded_attack_stems": payload["excluded_attack_stems"],
        "num_models": payload["num_models"],
        "per_sample_policy": payload["per_sample_policy"],
        "rows_with_computed_worstcase": payload["rows_with_computed_worstcase"],
        "rows": paper_rows,
    }

    write_json(output_dir / "all_attacks_summary.json", payload)
    write_csv(output_dir / "all_attacks_summary.csv", rows)
    write_json(output_dir / "all_attacks_worstcase_summary.json", paper_payload)
    write_csv(output_dir / "all_attacks_worstcase_summary.csv", paper_rows)

    table_rows = [[
        "model_id",
        "family",
        "variant",
        "threshold",
        "clean",
        *attack_stems,
        "mean",
        "worst",
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
        values.append(format_num(row["worst_imagewise_attack_miou"]))
        table_rows.append(values)

    paper_table_rows = [[
        "model_id",
        "family",
        "variant",
        "threshold",
        "clean",
        "attack_mean",
        "imagewise_worst",
        "worst_attack_mode",
        "counted",
    ]]
    for row in rows:
        paper_table_rows.append(
            [
                f"`{row['model_id']}`",
                f"`{row['family']}`",
                f"`{row['variant']}`",
                format_threshold(row["threshold"]),
                format_num(row["clean_miou"]),
                format_num(row["mean_attack_miou"]),
                format_num(row["worst_imagewise_attack_miou"]),
                row["worst_attack_stem_by_frequency"] or "-",
                str(row["num_attacks_counted"]),
            ]
        )

    table_lines = build_aligned_markdown_table(
        table_rows,
        right_align=set(range(3, len(table_rows[0]))),
    )
    paper_table_lines = build_aligned_markdown_table(
        paper_table_rows,
        right_align={3, 4, 5, 6, 8},
    )

    write_markdown(
        output_dir / "all_attacks_summary.md",
        "VOC All Attacks Summary",
        [
            f"- manifest: {manifest_path.resolve()}",
            f"- suite_root: {suite_root.resolve()}",
            f"- num_models: {len(rows)}",
            f"- attacks: {', '.join(attack_stems)}",
            f"- per_sample_policy: {args.per_sample_policy}",
            "",
            *table_lines,
        ],
    )
    write_markdown(
        output_dir / "all_attacks_worstcase_summary.md",
        "VOC All Attacks Worst-Case Summary",
        [
            f"- manifest: {manifest_path.resolve()}",
            f"- suite_root: {suite_root.resolve()}",
            f"- num_models: {len(rows)}",
            f"- attacks: {', '.join(attack_stems)}",
            f"- per_sample_policy: {args.per_sample_policy}",
            f"- rows_with_computed_worstcase: {payload['rows_with_computed_worstcase']}",
            "",
            *paper_table_lines,
        ],
    )
    print(output_dir / "all_attacks_summary.json")


if __name__ == "__main__":
    main()
