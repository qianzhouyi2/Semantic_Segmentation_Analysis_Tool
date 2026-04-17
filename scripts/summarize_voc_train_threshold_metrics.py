from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize train-split sparse threshold search metrics with optional baseline rows.")
    parser.add_argument("--search-root", required=True, help="Root directory containing */*/search_summary.json files.")
    parser.add_argument(
        "--baseline-root",
        default="",
        help="Optional root directory containing train-split baseline clean/pgd summaries.",
    )
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to <search-root>/summary.")
    parser.add_argument(
        "--output-name",
        default="train_split_best_threshold_vs_baseline_metrics",
        help="Basename for generated JSON/CSV/Markdown outputs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_num(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def build_aligned_markdown_table(rows: list[list[str]], *, right_align: set[int]) -> list[str]:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(row))) + " |"

    def sep_cell(index: int) -> str:
        width = max(3, widths[index])
        return ("-" * (width - 1) + ":") if index in right_align else ("-" * width)

    return [
        fmt(rows[0]),
        "| " + " | ".join(sep_cell(index) for index in range(len(widths))) + " |",
        *(fmt(row) for row in rows[1:]),
    ]


def collect_sparse_rows(search_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(search_root.glob("*/*/search_summary.json")):
        payload = load_json(path)
        best = payload["best_threshold"]
        rows.append(
            {
                "checkpoint": Path(payload["checkpoint"]).stem,
                "family": payload["family"],
                "variant": payload["variant"],
                "best_threshold": float(best["threshold"]),
                "clean_aacc": None if best.get("clean_aacc") is None else float(best["clean_aacc"]),
                "clean_miou": None if best.get("clean_miou") is None else float(best["clean_miou"]),
                "pgd_miou": float(best["adv_miou"]),
                "source": str(path.resolve()),
            }
        )
    return rows


def collect_baseline_rows(baseline_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    clean_root = baseline_root / "clean"
    pgd_root = baseline_root / "attacks" / "pgd"
    if not clean_root.exists() or not pgd_root.exists():
        return rows

    for clean_summary_path in sorted(clean_root.glob("baseline__*/summary.json")):
        model_id = clean_summary_path.parent.name
        pgd_summary_path = pgd_root / model_id / "summary.json"
        if not pgd_summary_path.exists():
            continue
        clean_summary = load_json(clean_summary_path)
        pgd_summary = load_json(pgd_summary_path)
        model = clean_summary["model"]
        rows.append(
            {
                "checkpoint": Path(model["checkpoint"]).stem,
                "family": model["family"],
                "variant": "baseline",
                "best_threshold": None,
                "clean_aacc": float(clean_summary["reference_percent"]["aAcc"]),
                "clean_miou": float(clean_summary["reference_percent"]["mIoU"]),
                "pgd_miou": float(pgd_summary["reference_percent"]["mIoU"]),
                "source": str(clean_summary_path.resolve()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    search_root = Path(args.search_root)
    baseline_root = Path(args.baseline_root) if args.baseline_root else None
    output_dir = Path(args.output_dir) if args.output_dir else search_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_sparse_rows(search_root)
    if baseline_root is not None:
        rows.extend(collect_baseline_rows(baseline_root))

    rows.sort(key=lambda row: (str(row["checkpoint"]), str(row["variant"])))
    payload = {
        "search_root": str(search_root.resolve()),
        "baseline_root": None if baseline_root is None else str(baseline_root.resolve()),
        "num_cases": len(rows),
        "rows": rows,
    }

    json_path = output_dir / f"{args.output_name}.json"
    csv_path = output_dir / f"{args.output_name}.csv"
    md_path = output_dir / f"{args.output_name}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    header = ["checkpoint", "family", "variant", "best_threshold", "clean_aAcc", "clean_mIoU", "PGD mIoU"]
    table_rows = [header]
    for row in rows:
        table_rows.append(
            [
                f"`{row['checkpoint']}`",
                f"`{row['family']}`",
                f"`{row['variant']}`",
                format_num(row["best_threshold"]),
                format_num(row["clean_aacc"]),
                format_num(row["clean_miou"]),
                format_num(row["pgd_miou"]),
            ]
        )

    lines = [
        "# Train-Split Best Threshold Metrics",
        "",
        f"- search_root: {search_root.resolve()}",
        (
            "- baseline_root: <none>"
            if baseline_root is None
            else f"- baseline_root: {baseline_root.resolve()}"
        ),
        f"- num_cases: {len(rows)}",
        "- note: `clean_aAcc` is train-split reference `aAcc`; baseline rows use train-split clean/PGD evaluation without sparse defense.",
        "",
        *build_aligned_markdown_table(table_rows, right_align={3, 4, 5, 6}),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json_path)


if __name__ == "__main__":
    main()
