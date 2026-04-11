from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.reporting.exporter import write_json, write_markdown
from src.robustness.evaluation import compare_clean_and_adversarial
from src.robustness.reporting import build_robustness_report_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare clean and adversarial segmentation summaries.")
    parser.add_argument("--clean-summary", required=True, help="Clean evaluation summary.json")
    parser.add_argument("--adv-summary", required=True, help="Adversarial evaluation summary.json")
    parser.add_argument("--output-dir", default="results/reports/robustness", help="Output directory.")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    clean_summary = _load_json(args.clean_summary)
    adv_summary = _load_json(args.adv_summary)
    summary = compare_clean_and_adversarial(clean_summary, adv_summary)

    output_dir = Path(args.output_dir)
    write_json(output_dir / "summary.json", summary.to_dict())
    write_markdown(output_dir / "report.md", "Robustness Evaluation", build_robustness_report_lines(summary))
    print(f"Robustness report written to: {output_dir}")


if __name__ == "__main__":
    main()
