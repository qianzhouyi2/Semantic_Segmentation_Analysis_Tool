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
    VOC_BASE_MODELS,
    VOC_DEFAULT_TRANSFER_ATTACK_STEMS,
    VOC_TRANSFER_ATTACKS,
    resolve_transfer_attacks,
)
from src.reporting.exporter import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize strict transfer-protocol experiment manifest.")
    parser.add_argument("--search-root", required=True, help="Threshold-search root with search_summary.json files.")
    parser.add_argument("--output-dir", required=True, help="Transfer-protocol output directory.")
    parser.add_argument(
        "--variants",
        default="meansparse,extrasparse",
        help="Comma or space separated sparse variants to include, or `all` for every sparse defense variant.",
    )
    parser.add_argument(
        "--attack-stems",
        default=",".join(VOC_DEFAULT_TRANSFER_ATTACK_STEMS),
        help=(
            "Comma or space separated transfer attack config stems. "
            "Defaults to the legacy subset: mi_fgsm, ni_di_ti. "
            "You can extend it with transegpgd or tass."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_attack_stems(raw: str | None) -> list[str]:
    if raw is None:
        return list(VOC_DEFAULT_TRANSFER_ATTACK_STEMS)
    stems = [item.strip() for chunk in raw.split(",") for item in chunk.split() if item.strip()]
    return stems or list(VOC_DEFAULT_TRANSFER_ATTACK_STEMS)


def build_sparse_config_lookup(
    search_root: Path,
    config_dir: Path,
    variants: list[str],
) -> tuple[dict[tuple[str, str], dict], list[Path]]:
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
        }
    return by_key, written_paths


def build_models(search_root: Path, output_dir: Path, variants: list[str]) -> tuple[list[dict], list[Path]]:
    config_dir = output_dir / "defense_configs"
    sparse_by_key, written_configs = build_sparse_config_lookup(search_root, config_dir, variants)
    models: list[dict] = []
    for base_model in VOC_BASE_MODELS:
        checkpoint_path = str((Path.cwd() / base_model["checkpoint"]).resolve())
        base_entry = {
            "model_id": f"baseline__{base_model['name']}",
            "display_name": base_model["name"],
            "family": base_model["family"],
            "regime": base_model["regime"],
            "checkpoint": checkpoint_path,
            "variant": "baseline",
            "defense_config": None,
            "threshold": None,
        }
        models.append(base_entry)
        for variant in variants:
            sparse_info = sparse_by_key.get((base_model["name"], variant))
            if sparse_info is None:
                raise FileNotFoundError(f"Missing threshold-search summary for {base_model['name']} {variant}")
            models.append(
                {
                    "model_id": f"{variant}__{base_model['name']}",
                    "display_name": base_model["name"],
                    "family": base_model["family"],
                    "regime": base_model["regime"],
                    "checkpoint": checkpoint_path,
                    "variant": variant,
                    "defense_config": sparse_info["config_path"],
                    "threshold": sparse_info["threshold"],
                }
            )
    return models, written_configs


def build_cases(
    models: list[dict],
    variants: list[str],
    transfer_attacks: list[dict] | tuple[dict, ...] = VOC_TRANSFER_ATTACKS,
) -> list[dict]:
    cases: list[dict] = []
    baseline_sources = [item for item in models if item["variant"] == "baseline"]
    grouped_targets: dict[tuple[str, str], list[dict]] = {}
    for item in models:
        grouped_targets.setdefault((item["display_name"], item["regime"]), []).append(item)
    variant_order = {variant: index for index, variant in enumerate(["baseline", *variants])}
    for group in grouped_targets.values():
        group.sort(key=lambda row: variant_order[row["variant"]])

    for attack in transfer_attacks:
        for source in baseline_sources:
            for (display_name, regime), targets in grouped_targets.items():
                if regime != source["regime"]:
                    continue
                relation = "same_family" if targets[0]["family"] == source["family"] else "cross_family"
                case_id = f"{attack['stem']}__{source['model_id']}__to__{display_name}"
                cases.append(
                    {
                        "case_id": case_id,
                        "attack_stem": attack["stem"],
                        "attack_name": attack["name"],
                        "attack_config": str((Path.cwd() / attack["config"]).resolve()),
                        "regime": regime,
                        "relation": relation,
                        "source": source,
                        "targets": targets,
                    }
                )
    return cases


def main() -> None:
    args = parse_args()
    search_root = Path(args.search_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = parse_sparse_variants(args.variants)
    attack_stems = parse_attack_stems(args.attack_stems)
    transfer_attacks = resolve_transfer_attacks(attack_stems)

    models, written_configs = build_models(search_root, output_dir, variants)
    cases = build_cases(models, variants, transfer_attacks=transfer_attacks)
    manifest = {
        "search_root": str(search_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "requested_sparse_variants": variants,
        "requested_attack_stems": attack_stems,
        "num_models": len(models),
        "num_cases": len(cases),
        "transfer_attacks": transfer_attacks,
        "models": models,
        "cases": cases,
        "written_defense_configs": [str(path.resolve()) for path in written_configs],
    }
    write_json(output_dir / "transfer_protocol_manifest.json", manifest)
    write_markdown(
        output_dir / "transfer_protocol_manifest.md",
        "VOC Transfer Protocol Manifest",
        [
            f"- search_root: {search_root.resolve()}",
            f"- output_dir: {output_dir.resolve()}",
            f"- sparse_variants: {', '.join(variants)}",
            f"- transfer_attacks: {', '.join(attack['stem'] for attack in transfer_attacks)}",
            f"- num_models: {len(models)}",
            f"- num_cases: {len(cases)}",
            "",
            "| case_id | regime | relation | attack | source | targets |",
            "| --- | --- | --- | --- | --- | --- |",
            *[
                f"| `{case['case_id']}` | `{case['regime']}` | `{case['relation']}` | `{case['attack_stem']}` | "
                f"`{case['source']['model_id']}` | "
                f"`{', '.join(target['model_id'] for target in case['targets'])}` |"
                for case in cases
            ],
        ],
    )
    print(output_dir / "transfer_protocol_manifest.json")


if __name__ == "__main__":
    main()
