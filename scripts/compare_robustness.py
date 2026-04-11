from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.reporting.exporter import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two robustness summary files.")
    parser.add_argument("--baseline", required=True, help="Baseline robustness summary.json")
    parser.add_argument("--candidate", required=True, help="Candidate robustness summary.json")
    parser.add_argument("--output-dir", default="results/reports/robustness_comparison", help="Output directory.")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    baseline = _load_json(args.baseline)
    candidate = _load_json(args.candidate)
    delta = {key: float(candidate[key]) - float(baseline[key]) for key in baseline.keys() if key in candidate}

    output_dir = Path(args.output_dir)
    write_json(output_dir / "comparison.json", delta)
    write_markdown(
        output_dir / "report.md",
        "Robustness Comparison",
        [f"- {key}: {value:.4f}" for key, value in sorted(delta.items())],
    )
    print(f"Robustness comparison report written to: {output_dir}")


if __name__ == "__main__":
    main()
