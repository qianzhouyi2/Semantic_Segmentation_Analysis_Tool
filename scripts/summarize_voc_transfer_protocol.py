from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

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


def mean(values: Iterable[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / float(len(numeric))


def build_variant_order(manifest: dict) -> dict[str, int]:
    variants = ["baseline", *manifest.get("requested_sparse_variants", [])]
    return {variant: index for index, variant in enumerate(variants)}


def variant_sort_key(variant: str, variant_order: dict[str, int]) -> tuple[int, str]:
    return (variant_order.get(variant, len(variant_order)), variant)


def case_output_dir(suite_root: Path, case: dict) -> Path:
    return suite_root / "cases" / case["attack_stem"] / case["source"]["model_id"] / case["targets"][0]["display_name"]


def build_case_and_target_rows(manifest: dict, suite_root: Path) -> tuple[list[dict], list[dict], list[str]]:
    case_rows: list[dict] = []
    target_rows: list[dict] = []
    missing_case_ids: list[str] = []

    for case in manifest["cases"]:
        summary_path = case_output_dir(suite_root, case) / "summary.json"
        if not summary_path.exists():
            missing_case_ids.append(case["case_id"])
            continue

        payload = load_json(summary_path)
        target_manifest_by_id = {target["model_id"]: target for target in case["targets"]}
        baseline_target_payload = next(
            (item for item in payload["targets"] if target_manifest_by_id[item["model"]["model_id"]]["variant"] == "baseline"),
            None,
        )
        baseline_transfer_miou = None
        baseline_transfer_drop = None
        baseline_target_model_id = None
        if baseline_target_payload is not None:
            baseline_transfer_miou = float(baseline_target_payload["transfer"]["reference_percent"]["mIoU"])
            baseline_transfer_drop = float(baseline_target_payload["transfer_miou_drop"])
            baseline_target_model_id = str(baseline_target_payload["model"]["model_id"])

        case_rows.append(
            {
                "case_id": str(payload["case_id"]),
                "regime": str(payload["regime"]),
                "relation": str(payload["relation"]),
                "attack_stem": str(payload["attack"]["stem"]),
                "attack_name": str(payload["attack"]["name"]),
                "source_model_id": str(payload["source_self"]["model"]["model_id"]),
                "source_display_name": str(case["source"]["display_name"]),
                "source_family": str(payload["source_self"]["model"]["family"]),
                "source_variant": str(case["source"]["variant"]),
                "target_display_name": str(case["targets"][0]["display_name"]),
                "target_group_size": len(case["targets"]),
                "source_clean_miou": float(payload["source_self"]["clean"]["reference_percent"]["mIoU"]),
                "source_transfer_miou": float(payload["source_self"]["transfer"]["reference_percent"]["mIoU"]),
                "source_transfer_drop": float(payload["source_self"]["transfer_miou_drop"]),
                "summary_path": str(summary_path.resolve()),
            }
        )

        for target_payload in payload["targets"]:
            target_model_id = str(target_payload["model"]["model_id"])
            target_manifest = target_manifest_by_id[target_model_id]
            clean_miou = float(target_payload["clean"]["reference_percent"]["mIoU"])
            transfer_miou = float(target_payload["transfer"]["reference_percent"]["mIoU"])
            transfer_drop = float(target_payload["transfer_miou_drop"])
            gain_over_baseline_miou = None if baseline_transfer_miou is None else transfer_miou - baseline_transfer_miou
            drop_reduction_vs_baseline = (
                None if baseline_transfer_drop is None else baseline_transfer_drop - transfer_drop
            )
            target_rows.append(
                {
                    "case_id": str(payload["case_id"]),
                    "regime": str(payload["regime"]),
                    "relation": str(payload["relation"]),
                    "attack_stem": str(payload["attack"]["stem"]),
                    "attack_name": str(payload["attack"]["name"]),
                    "source_model_id": str(payload["source_self"]["model"]["model_id"]),
                    "source_display_name": str(case["source"]["display_name"]),
                    "source_family": str(payload["source_self"]["model"]["family"]),
                    "source_variant": str(case["source"]["variant"]),
                    "target_model_id": target_model_id,
                    "target_display_name": str(target_manifest["display_name"]),
                    "target_family": str(target_payload["model"]["family"]),
                    "target_variant": str(target_manifest["variant"]),
                    "target_threshold": target_manifest["threshold"],
                    "target_is_baseline": target_manifest["variant"] == "baseline",
                    "clean_miou": clean_miou,
                    "transfer_miou": transfer_miou,
                    "transfer_miou_drop": transfer_drop,
                    "baseline_target_model_id": baseline_target_model_id,
                    "baseline_transfer_miou": baseline_transfer_miou,
                    "baseline_transfer_miou_drop": baseline_transfer_drop,
                    "gain_over_baseline_miou": gain_over_baseline_miou,
                    "drop_reduction_vs_baseline": drop_reduction_vs_baseline,
                    "summary_path": str(summary_path.resolve()),
                }
            )

    return case_rows, target_rows, missing_case_ids


def sort_case_rows(case_rows: list[dict]) -> list[dict]:
    return sorted(
        case_rows,
        key=lambda row: (row["regime"], row["attack_stem"], row["source_model_id"], row["target_display_name"]),
    )


def sort_target_rows(target_rows: list[dict], variant_order: dict[str, int]) -> list[dict]:
    return sorted(
        target_rows,
        key=lambda row: (
            row["regime"],
            row["target_display_name"],
            variant_sort_key(row["target_variant"], variant_order),
            row["attack_stem"],
            row["source_model_id"],
        ),
    )


def build_source_strength_rows(case_rows: list[dict]) -> list[dict]:
    grouped_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in case_rows:
        grouped_rows[(row["attack_stem"], row["source_model_id"])].append(row)

    rows: list[dict] = []
    for (_, _), items in grouped_rows.items():
        first = items[0]
        rows.append(
            {
                "regime": first["regime"],
                "attack_stem": first["attack_stem"],
                "attack_name": first["attack_name"],
                "source_model_id": first["source_model_id"],
                "source_display_name": first["source_display_name"],
                "source_family": first["source_family"],
                "source_clean_miou": mean(row["source_clean_miou"] for row in items),
                "source_transfer_miou": mean(row["source_transfer_miou"] for row in items),
                "source_transfer_drop": mean(row["source_transfer_drop"] for row in items),
                "num_target_groups": len(items),
            }
        )
    return sorted(rows, key=lambda row: (row["regime"], row["attack_stem"], row["source_model_id"]))


def build_target_comparison_rows(
    target_rows: list[dict],
    *,
    relation: str,
    variant_order: dict[str, int],
) -> list[dict]:
    rows = [row for row in target_rows if row["relation"] == relation]
    return sorted(
        rows,
        key=lambda row: (
            row["regime"],
            row["target_display_name"],
            variant_sort_key(row["target_variant"], variant_order),
            row["attack_stem"],
            row["source_model_id"],
        ),
    )


def build_family_matrix_rows(target_rows: list[dict]) -> list[dict]:
    grouped_rows: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in target_rows:
        if row["target_variant"] != "baseline":
            continue
        grouped_rows[(row["regime"], row["attack_stem"], row["source_family"], row["target_family"])].append(row)

    rows: list[dict] = []
    for (_, _, _, _), items in grouped_rows.items():
        first = items[0]
        rows.append(
            {
                "regime": first["regime"],
                "attack_stem": first["attack_stem"],
                "source_family": first["source_family"],
                "target_family": first["target_family"],
                "num_pairs": len(items),
                "baseline_target_transfer_miou": mean(row["transfer_miou"] for row in items),
                "baseline_target_transfer_drop": mean(row["transfer_miou_drop"] for row in items),
                "target_model_ids": ", ".join(sorted(str(row["target_model_id"]) for row in items)),
                "source_model_ids": ", ".join(sorted(str(row["source_model_id"]) for row in items)),
            }
        )
    return sorted(rows, key=lambda row: (row["regime"], row["attack_stem"], row["source_family"], row["target_family"]))


def build_gain_over_baseline_rows(target_rows: list[dict], variant_order: dict[str, int]) -> list[dict]:
    rows = [row for row in target_rows if row["target_variant"] != "baseline"]
    return sorted(
        rows,
        key=lambda row: (
            row["regime"],
            row["target_display_name"],
            row["attack_stem"],
            row["source_model_id"],
            variant_sort_key(row["target_variant"], variant_order),
        ),
    )


def pick_worst_row(rows: list[dict]) -> dict:
    return min(
        rows,
        key=lambda row: (
            float(row["transfer_miou"]),
            float(row["transfer_miou_drop"]) * -1.0,
            str(row["attack_stem"]),
            str(row["source_model_id"]),
        ),
    )


def build_worstcase_payload(target_rows: list[dict], variant_order: dict[str, int]) -> dict:
    overall_groups: dict[str, list[dict]] = defaultdict(list)
    attack_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    source_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in target_rows:
        target_model_id = str(row["target_model_id"])
        overall_groups[target_model_id].append(row)
        attack_groups[(target_model_id, str(row["attack_stem"]))].append(row)
        source_groups[(target_model_id, str(row["source_model_id"]))].append(row)

    overall_rows: list[dict] = []
    per_attack_rows: list[dict] = []
    per_source_rows: list[dict] = []

    for target_model_id, items in overall_groups.items():
        worst = pick_worst_row(items)
        overall_rows.append(
            {
                "target_model_id": target_model_id,
                "target_display_name": worst["target_display_name"],
                "target_family": worst["target_family"],
                "target_variant": worst["target_variant"],
                "regime": worst["regime"],
                "clean_miou": mean(row["clean_miou"] for row in items),
                "worst_transfer_miou": float(worst["transfer_miou"]),
                "worst_transfer_miou_drop": float(worst["transfer_miou_drop"]),
                "worst_attack_stem": worst["attack_stem"],
                "worst_source_model_id": worst["source_model_id"],
                "worst_source_family": worst["source_family"],
                "worst_relation": worst["relation"],
                "baseline_target_model_id": worst["baseline_target_model_id"],
                "worst_gain_over_baseline_miou": worst["gain_over_baseline_miou"],
                "num_source_attack_pairs": len(items),
                "worst_case_id": worst["case_id"],
            }
        )

    for (target_model_id, attack_stem), items in attack_groups.items():
        worst = pick_worst_row(items)
        per_attack_rows.append(
            {
                "target_model_id": target_model_id,
                "target_display_name": worst["target_display_name"],
                "target_variant": worst["target_variant"],
                "regime": worst["regime"],
                "attack_stem": attack_stem,
                "worst_source_model_id": worst["source_model_id"],
                "worst_source_family": worst["source_family"],
                "worst_relation": worst["relation"],
                "worst_transfer_miou": float(worst["transfer_miou"]),
                "worst_transfer_miou_drop": float(worst["transfer_miou_drop"]),
                "num_sources": len(items),
                "worst_case_id": worst["case_id"],
            }
        )

    for (target_model_id, source_model_id), items in source_groups.items():
        worst = pick_worst_row(items)
        per_source_rows.append(
            {
                "target_model_id": target_model_id,
                "target_display_name": worst["target_display_name"],
                "target_variant": worst["target_variant"],
                "regime": worst["regime"],
                "source_model_id": source_model_id,
                "source_family": worst["source_family"],
                "worst_attack_stem": worst["attack_stem"],
                "worst_relation": worst["relation"],
                "worst_transfer_miou": float(worst["transfer_miou"]),
                "worst_transfer_miou_drop": float(worst["transfer_miou_drop"]),
                "num_attacks": len(items),
                "worst_case_id": worst["case_id"],
            }
        )

    overall_rows = sorted(
        overall_rows,
        key=lambda row: (
            row["regime"],
            row["target_display_name"],
            variant_sort_key(str(row["target_variant"]), variant_order),
        ),
    )
    per_attack_rows = sorted(
        per_attack_rows,
        key=lambda row: (
            row["regime"],
            row["target_display_name"],
            variant_sort_key(str(row["target_variant"]), variant_order),
            row["attack_stem"],
        ),
    )
    per_source_rows = sorted(
        per_source_rows,
        key=lambda row: (
            row["regime"],
            row["target_display_name"],
            variant_sort_key(str(row["target_variant"]), variant_order),
            row["source_model_id"],
        ),
    )
    return {
        "per_target_worst_overall": overall_rows,
        "per_target_worst_by_attack": per_attack_rows,
        "per_target_worst_by_source": per_source_rows,
    }


def render_table(
    rows: list[dict],
    columns: list[tuple[str, str]],
    *,
    empty_message: str = "- <none>",
) -> list[str]:
    if not rows:
        return [empty_message]

    def render_value(value: object) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---:" if key.endswith("miou") or key.endswith("drop") else "---" for key, _ in columns) + " |"
    body = ["| " + " | ".join(render_value(row.get(key)) for key, _ in columns) + " |" for row in rows]
    return [header, separator, *body]


def build_tables_payload(case_rows: list[dict], target_rows: list[dict], variant_order: dict[str, int]) -> tuple[dict, dict]:
    source_strength_rows = build_source_strength_rows(case_rows)
    same_family_rows = build_target_comparison_rows(target_rows, relation="same_family", variant_order=variant_order)
    cross_family_rows = build_target_comparison_rows(target_rows, relation="cross_family", variant_order=variant_order)
    family_matrix_rows = build_family_matrix_rows(target_rows)
    gain_over_baseline_rows = build_gain_over_baseline_rows(target_rows, variant_order)
    worstcase_payload = build_worstcase_payload(target_rows, variant_order)
    tables = {
        "source_self_transfer_strength": source_strength_rows,
        "fixed_target_same_family_comparison": same_family_rows,
        "fixed_target_cross_family_comparison": cross_family_rows,
        "source_target_family_matrix_on_baseline_targets": family_matrix_rows,
        "gain_over_baseline": gain_over_baseline_rows,
    }
    csv_tables = {
        "transfer_protocol_source_strength.csv": source_strength_rows,
        "transfer_protocol_same_family.csv": same_family_rows,
        "transfer_protocol_cross_family.csv": cross_family_rows,
        "transfer_protocol_family_matrix.csv": family_matrix_rows,
        "transfer_protocol_gain_over_baseline.csv": gain_over_baseline_rows,
    }
    return tables | {"worstcase_overview": worstcase_payload["per_target_worst_overall"]}, {
        "csv_tables": csv_tables,
        "worstcase_payload": worstcase_payload,
    }


def build_tables_markdown(
    *,
    manifest_path: Path,
    suite_root: Path,
    case_rows: list[dict],
    target_rows: list[dict],
    tables: dict,
    worstcase_payload: dict,
) -> list[str]:
    return [
        f"- manifest: {manifest_path.resolve()}",
        f"- suite_root: {suite_root.resolve()}",
        f"- num_cases: {len(case_rows)}",
        f"- num_target_rows: {len(target_rows)}",
        "",
        "## Source Self Transfer Strength",
        *render_table(
            tables["source_self_transfer_strength"],
            [
                ("regime", "regime"),
                ("attack_stem", "attack"),
                ("source_model_id", "source"),
                ("source_clean_miou", "clean mIoU"),
                ("source_transfer_miou", "self transfer mIoU"),
                ("source_transfer_drop", "self drop"),
                ("num_target_groups", "target groups"),
            ],
        ),
        "",
        "## Fixed Target Same-Family Comparison",
        *render_table(
            tables["fixed_target_same_family_comparison"],
            [
                ("regime", "regime"),
                ("attack_stem", "attack"),
                ("source_model_id", "source"),
                ("target_model_id", "target"),
                ("clean_miou", "clean mIoU"),
                ("transfer_miou", "transfer mIoU"),
                ("transfer_miou_drop", "drop"),
                ("gain_over_baseline_miou", "gain vs baseline"),
            ],
        ),
        "",
        "## Fixed Target Cross-Family Comparison",
        *render_table(
            tables["fixed_target_cross_family_comparison"],
            [
                ("regime", "regime"),
                ("attack_stem", "attack"),
                ("source_model_id", "source"),
                ("target_model_id", "target"),
                ("clean_miou", "clean mIoU"),
                ("transfer_miou", "transfer mIoU"),
                ("transfer_miou_drop", "drop"),
                ("gain_over_baseline_miou", "gain vs baseline"),
            ],
        ),
        "",
        "## Source-Target Family Matrix On Baseline Targets",
        *render_table(
            tables["source_target_family_matrix_on_baseline_targets"],
            [
                ("regime", "regime"),
                ("attack_stem", "attack"),
                ("source_family", "source family"),
                ("target_family", "target family"),
                ("baseline_target_transfer_miou", "baseline transfer mIoU"),
                ("baseline_target_transfer_drop", "baseline drop"),
                ("num_pairs", "pairs"),
            ],
        ),
        "",
        "## Gain Over Baseline",
        *render_table(
            tables["gain_over_baseline"],
            [
                ("regime", "regime"),
                ("attack_stem", "attack"),
                ("source_model_id", "source"),
                ("target_model_id", "target"),
                ("baseline_target_model_id", "baseline target"),
                ("transfer_miou", "transfer mIoU"),
                ("baseline_transfer_miou", "baseline transfer"),
                ("gain_over_baseline_miou", "gain vs baseline"),
                ("drop_reduction_vs_baseline", "drop reduction"),
            ],
        ),
        "",
        "## Worst-Case Target Summary",
        *render_table(
            worstcase_payload["per_target_worst_overall"],
            [
                ("regime", "regime"),
                ("target_model_id", "target"),
                ("worst_attack_stem", "worst attack"),
                ("worst_source_model_id", "worst source"),
                ("worst_transfer_miou", "worst transfer mIoU"),
                ("worst_transfer_miou_drop", "worst drop"),
                ("worst_gain_over_baseline_miou", "gain vs baseline"),
            ],
        ),
    ]


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    suite_root = Path(args.suite_root)
    manifest = load_json(manifest_path)
    variant_order = build_variant_order(manifest)

    case_rows, target_rows, missing_case_ids = build_case_and_target_rows(manifest, suite_root)
    case_rows = sort_case_rows(case_rows)
    target_rows = sort_target_rows(target_rows, variant_order)

    tables, extras = build_tables_payload(case_rows, target_rows, variant_order)
    worstcase_payload = extras["worstcase_payload"]

    output_dir = suite_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": str(manifest_path.resolve()),
        "suite_root": str(suite_root.resolve()),
        "requested_attack_stems": manifest.get("requested_attack_stems", [attack["stem"] for attack in manifest["transfer_attacks"]]),
        "transfer_attacks": manifest["transfer_attacks"],
        "num_manifest_cases": int(manifest["num_cases"]),
        "num_cases": len(case_rows),
        "num_target_rows": len(target_rows),
        "missing_case_ids": missing_case_ids,
        "case_rows": case_rows,
        "target_rows": target_rows,
        "tables": tables,
    }
    write_json(output_dir / "transfer_protocol_summary.json", payload)
    write_csv(output_dir / "transfer_protocol_cases.csv", case_rows)
    write_csv(output_dir / "transfer_protocol_targets.csv", target_rows)
    for filename, rows in extras["csv_tables"].items():
        write_csv(output_dir / filename, rows)
    write_json(output_dir / "transfer_protocol_worstcase.json", worstcase_payload)
    write_csv(output_dir / "transfer_protocol_worstcase.csv", worstcase_payload["per_target_worst_overall"])

    write_markdown(
        output_dir / "transfer_protocol_summary.md",
        "VOC Transfer Protocol Summary",
        [
            f"- manifest: {manifest_path.resolve()}",
            f"- suite_root: {suite_root.resolve()}",
            f"- requested_attack_stems: {', '.join(payload['requested_attack_stems'])}",
            f"- num_cases: {len(case_rows)} / {manifest['num_cases']}",
            f"- num_target_rows: {len(target_rows)}",
            f"- missing_case_ids: {', '.join(missing_case_ids) if missing_case_ids else '<none>'}",
            "",
            "| case_id | regime | relation | attack | source | target group | source clean | source transfer | source drop |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
            *[
                f"| `{row['case_id']}` | `{row['regime']}` | `{row['relation']}` | `{row['attack_stem']}` | "
                f"`{row['source_model_id']}` | `{row['target_display_name']}` | "
                f"{fmt(row['source_clean_miou'])} | {fmt(row['source_transfer_miou'])} | {fmt(row['source_transfer_drop'])} |"
                for row in case_rows
            ],
        ],
    )
    write_markdown(
        output_dir / "transfer_protocol_tables.md",
        "VOC Transfer Protocol Tables",
        build_tables_markdown(
            manifest_path=manifest_path,
            suite_root=suite_root,
            case_rows=case_rows,
            target_rows=target_rows,
            tables=tables,
            worstcase_payload=worstcase_payload,
        ),
    )
    print(output_dir / "transfer_protocol_summary.json")


if __name__ == "__main__":
    main()
