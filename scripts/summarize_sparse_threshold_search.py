from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.reporting.exporter import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sparse threshold search results.")
    parser.add_argument("--search-root", required=True, help="Root directory containing per-case search_summary.json files.")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to search root.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_clean_miou(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def repo_relative_path(target: Path, repo_root: Path) -> str:
    try:
        return str(target.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(target.resolve())


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    search_root = Path(args.search_root)
    output_dir = Path(args.output_dir) if args.output_dir else search_root
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for summary_path in sorted(search_root.glob("*/*/search_summary.json")):
        payload = load_json(summary_path)
        best = payload["best_threshold"]
        rows.append(
            {
                "family": payload["family"],
                "checkpoint": repo_relative_path(Path(payload["checkpoint"]), repo_root),
                "variant": payload["variant"],
                "best_threshold": float(best["threshold"]),
                "clean_miou": None if best["clean_miou"] is None else float(best["clean_miou"]),
                "adv_miou": float(best["adv_miou"]),
                "search_summary": repo_relative_path(summary_path, repo_root),
            }
        )

    summary = {
        "search_root": repo_relative_path(search_root, repo_root),
        "num_cases": len(rows),
        "rows": rows,
    }
    write_json(output_dir / "selected_thresholds.json", summary)
    write_markdown(
        output_dir / "selected_thresholds.md",
        "Sparse Threshold Search Summary",
        [
            f"- search_root: {repo_relative_path(search_root, repo_root)}",
            f"- num_cases: {len(rows)}",
            "",
            "| checkpoint | family | variant | best threshold | clean mIoU | PGD mIoU |",
            "| --- | --- | --- | ---: | ---: | ---: |",
            *[
                f"| `{Path(row['checkpoint']).name}` | `{row['family']}` | `{row['variant']}` | {row['best_threshold']:.2f} | "
                f"{format_clean_miou(row['clean_miou'])} | {row['adv_miou']:.2f} |"
                for row in rows
            ],
        ],
    )
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
