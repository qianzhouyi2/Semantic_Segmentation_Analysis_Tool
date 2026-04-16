from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from src.reporting.exporter import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize sparse defense YAML configs from threshold-search summaries."
    )
    parser.add_argument("--search-root", required=True, help="Root directory containing */*/search_summary.json files.")
    parser.add_argument("--output-dir", default="configs/defenses", help="Directory to write generated YAML configs.")
    parser.add_argument(
        "--summary-name",
        default="selected_sparse_defense_configs",
        help="Basename for generated JSON/Markdown summary files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relativize_path(target: Path, base_dir: Path) -> str:
    try:
        return os.path.relpath(target, start=base_dir)
    except ValueError:
        return str(target)


def repo_relative_path(target: Path, repo_root: Path) -> str:
    try:
        return os.path.relpath(target.resolve(), start=repo_root.resolve())
    except ValueError:
        return str(target.resolve())


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    search_root = Path(args.search_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    written_paths: list[Path] = []
    for summary_path in sorted(search_root.glob("*/*/search_summary.json")):
        payload = load_json(summary_path)
        checkpoint_path = Path(payload["checkpoint"])
        checkpoint_name = checkpoint_path.stem
        family = str(payload["family"])
        variant = str(payload["variant"])
        best_threshold = float(payload["best_threshold"]["threshold"])
        stats_path = Path(payload["stats_path"])

        config_path = output_dir / f"{checkpoint_name}_{variant}.yaml"
        config_payload = {
            "name": variant,
            "family": family,
            "threshold": best_threshold,
            "stats_path": relativize_path(stats_path.resolve(), config_path.parent.resolve()),
            "strict_stats": True,
        }
        config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
        written_paths.append(config_path)
        rows.append(
            {
                "checkpoint": repo_relative_path(checkpoint_path, repo_root),
                "checkpoint_name": checkpoint_name,
                "family": family,
                "variant": variant,
                "threshold": best_threshold,
                "stats_path": repo_relative_path(stats_path, repo_root),
                "config_path": repo_relative_path(config_path, repo_root),
                "search_summary": repo_relative_path(summary_path, repo_root),
            }
        )

    summary = {
        "search_root": repo_relative_path(search_root, repo_root),
        "output_dir": repo_relative_path(output_dir, repo_root),
        "num_configs": len(rows),
        "rows": rows,
    }
    write_json(output_dir / f"{args.summary_name}.json", summary)
    write_markdown(
        output_dir / f"{args.summary_name}.md",
        "Selected Sparse Defense Configs",
        [
            f"- search_root: {repo_relative_path(search_root, repo_root)}",
            f"- output_dir: {repo_relative_path(output_dir, repo_root)}",
            f"- num_configs: {len(rows)}",
            "",
            "| checkpoint | family | variant | threshold | config |",
            "| --- | --- | --- | ---: | --- |",
            *[
                f"| `{row['checkpoint_name']}` | `{row['family']}` | `{row['variant']}` | {row['threshold']:.2f} | "
                f"`{Path(str(row['config_path'])).name}` |"
                for row in rows
            ],
        ],
    )
    print(output_dir.resolve())
    for path in written_paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
