from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from src.common.sparse_workflow import (
    parse_sparse_variants,
    resolve_sparse_config_from_search_summary,
    serialize_sparse_defense_config,
)
from src.common.voc_protocol import (
    VOC_ATTACK_SUITE_TRANSFER_SOURCE_BY_FAMILY,
    VOC_BASE_MODELS,
    VOC_DEFAULT_TRANSFER_ATTACK_STEMS,
)
from src.reporting.exporter import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a VOC attack-suite manifest from sparse threshold-search results.")
    parser.add_argument("--search-root", required=True, help="Threshold-search root containing per-case search_summary.json.")
    parser.add_argument("--output-dir", required=True, help="Attack-suite output directory.")
    parser.add_argument(
        "--variants",
        default="meansparse,extrasparse",
        help="Comma or space separated sparse variants to include, or `all` for every sparse defense variant.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_threshold(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def build_sparse_configs(search_root: Path, config_dir: Path, variants: list[str]) -> tuple[dict[tuple[str, str], dict], list[Path]]:
    by_key: dict[tuple[str, str], dict] = {}
    written_paths: list[Path] = []
    for summary_path in sorted(search_root.glob("*/*/search_summary.json")):
        payload = load_json(summary_path)
        checkpoint_name = Path(payload["checkpoint"]).stem
        variant = str(payload["variant"])
        if variant not in variants:
            continue
        config = resolve_sparse_config_from_search_summary(payload, summary_path=summary_path)
        config_path = config_dir / f"{checkpoint_name}_{variant}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_payload = serialize_sparse_defense_config(config, relative_to=config_path.parent.resolve())
        config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
        written_paths.append(config_path)
        by_key[(checkpoint_name, variant)] = {
            "config_path": str(config_path.resolve()),
            "threshold": float(config.threshold),
            "stats_path": None if config.stats_path is None else str(config.stats_path.resolve()),
            "clean_summary": str(Path(payload["best_threshold"]["clean_results"]).resolve()),
            "pgd_summary": str(Path(payload["best_threshold"]["adv_results"]).resolve()),
            "variant": variant,
        }
    return by_key, written_paths


def build_manifest(search_root: Path, output_dir: Path, variants: list[str]) -> dict:
    config_dir = output_dir / "defense_configs"
    sparse_by_key, written_configs = build_sparse_configs(search_root, config_dir, variants)

    models: list[dict] = []
    for base_model in VOC_BASE_MODELS:
        checkpoint_name = base_model["name"]
        checkpoint_path = str((Path.cwd() / base_model["checkpoint"]).resolve())

        baseline_entry = {
            "model_id": f"baseline__{checkpoint_name}",
            "display_name": checkpoint_name,
            "family": base_model["family"],
            "checkpoint": checkpoint_path,
            "variant": "baseline",
            "defense_config": None,
            "threshold": None,
            "reused_clean_summary": None,
            "reused_pgd_summary": None,
        }
        models.append(baseline_entry)

        for variant in variants:
            sparse_info = sparse_by_key.get((checkpoint_name, variant))
            if sparse_info is None:
                raise FileNotFoundError(f"Missing threshold-search summary for {checkpoint_name} {variant}")
            models.append(
                {
                    "model_id": f"{variant}__{checkpoint_name}",
                    "display_name": checkpoint_name,
                    "family": base_model["family"],
                    "checkpoint": checkpoint_path,
                    "variant": variant,
                    "defense_config": sparse_info["config_path"],
                    "threshold": sparse_info["threshold"],
                    "reused_clean_summary": sparse_info["clean_summary"],
                    "reused_pgd_summary": sparse_info["pgd_summary"],
                }
            )

    model_index = {item["model_id"]: item for item in models}
    for item in models:
        source_model_id = VOC_ATTACK_SUITE_TRANSFER_SOURCE_BY_FAMILY[item["family"]]
        item["transfer_source_model_id"] = source_model_id
        source = model_index[source_model_id]
        item["transfer_source_family"] = source["family"]
        item["transfer_source_checkpoint"] = source["checkpoint"]
        item["transfer_source_defense_config"] = source["defense_config"]

    manifest = {
        "search_root": str(search_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "requested_sparse_variants": variants,
        "num_models": len(models),
        "whitebox_attacks": ["pgd", "segpgd"],
        "blackbox_transfer_attacks": list(VOC_DEFAULT_TRANSFER_ATTACK_STEMS),
        "models": models,
        "written_defense_configs": [str(path.resolve()) for path in written_configs],
    }
    return manifest


def main() -> None:
    args = parse_args()
    search_root = Path(args.search_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = parse_sparse_variants(args.variants)

    manifest = build_manifest(search_root, output_dir, variants)
    write_json(output_dir / "attack_suite_manifest.json", manifest)
    write_markdown(
        output_dir / "attack_suite_manifest.md",
        "VOC Attack Suite Manifest",
        [
            f"- search_root: {search_root.resolve()}",
            f"- output_dir: {output_dir.resolve()}",
            f"- sparse_variants: {', '.join(variants)}",
            f"- num_models: {manifest['num_models']}",
            f"- whitebox_attacks: {', '.join(manifest['whitebox_attacks'])}",
            f"- blackbox_transfer_attacks: {', '.join(manifest['blackbox_transfer_attacks'])}",
            "",
            "| model_id | family | variant | threshold | transfer source |",
            "| --- | --- | --- | ---: | --- |",
            *[
                f"| `{row['model_id']}` | `{row['family']}` | `{row['variant']}` | "
                f"{format_threshold(row['threshold'])} | `{row['transfer_source_model_id']}` |"
                for row in manifest["models"]
            ],
        ],
    )
    print(output_dir / "attack_suite_manifest.json")


if __name__ == "__main__":
    main()
