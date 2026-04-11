from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.reporting.exporter import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a simple Markdown report from a JSON summary.")
    parser.add_argument("--summary-json", required=True, help="Input summary JSON file.")
    parser.add_argument("--output-md", required=True, help="Output markdown path.")
    parser.add_argument("--title", default="Analysis Report", help="Markdown title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.summary_json).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    lines = ["## Summary Payload", "```json", json.dumps(payload, indent=2, ensure_ascii=False), "```"]
    write_markdown(args.output_md, args.title, lines)
    print(f"Markdown report written to: {args.output_md}")


if __name__ == "__main__":
    main()
