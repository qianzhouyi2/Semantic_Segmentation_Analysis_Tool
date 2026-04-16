from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from src.reporting.exporter import write_json, write_markdown


BASE_MODELS = [
    {
        "name": "UperNet_ConvNext_T_VOC_adv",
        "family": "upernet_convnext",
        "checkpoint": "models/UperNet_ConvNext_T_VOC_adv.pth",
        "regime": "adv",
    },
    {
        "name": "UperNet_ConvNext_T_VOC_clean",
        "family": "upernet_convnext",
        "checkpoint": "models/UperNet_ConvNext_T_VOC_clean.pth",
        "regime": "clean",
    },
    {
        "name": "UperNet_ResNet50_VOC_adv",
        "family": "upernet_resnet50",
        "checkpoint": "models/UperNet_ResNet50_VOC_adv.pth",
        "regime": "adv",
    },
    {
        "name": "UperNet_ResNet50_VOC_clean",
        "family": "upernet_resnet50",
        "checkpoint": "models/UperNet_ResNet50_VOC_clean.pth",
        "regime": "clean",
    },
    {
        "name": "Segmenter_ViT_S_VOC_adv",
        "family": "segmenter_vit_s",
        "checkpoint": "models/Segmenter_ViT_S_VOC_adv.pth",
        "regime": "adv",
    },
    {
        "name": "Segmenter_ViT_S_VOC_clean",
        "family": "segmenter_vit_s",
        "checkpoint": "models/Segmenter_ViT_S_VOC_clean.pth",
        "regime": "clean",
    },
]

SPARSE_VARIANTS = ("meansparse", "extrasparse")
TRANSFER_ATTACKS = (
    {"stem": "mi_fgsm", "name": "mi-fgsm", "config": "configs/attacks/mi_fgsm.yaml"},
    {"stem": "ni_di_ti", "name": "ni+di+ti", "config": "configs/attacks/ni_di_ti.yaml"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize strict transfer-protocol experiment manifest.")
    parser.add_argument("--search-root", required=True, help="Threshold-search root with search_summary.json files.")
    parser.add_argument("--output-dir", required=True, help="Transfer-protocol output directory.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_name_to_model(name: str) -> dict:
    for item in BASE_MODELS:
        if item["name"] == name:
            return dict(item)
    raise KeyError(f"Unknown base model name in search summary: {name}")


def build_sparse_config_lookup(search_root: Path, config_dir: Path) -> tuple[dict[tuple[str, str], dict], list[Path]]:
    by_key: dict[tuple[str, str], dict] = {}
    written_paths: list[Path] = []
    for summary_path in sorted(search_root.glob("*/*/search_summary.json")):
        payload = load_json(summary_path)
        checkpoint_name = Path(payload["checkpoint"]).stem
        base_model = checkpoint_name_to_model(checkpoint_name)
        variant = str(payload["variant"])
        threshold = float(payload["best_threshold"]["threshold"])
        stats_path = str(payload["stats_path"])
        config_payload = {
            "name": variant,
            "family": base_model["family"],
            "threshold": threshold,
            "stats_path": stats_path,
            "strict_stats": True,
        }
        config_path = config_dir / f"{checkpoint_name}_{variant}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
        written_paths.append(config_path)
        by_key[(checkpoint_name, variant)] = {
            "config_path": str(config_path.resolve()),
            "threshold": threshold,
            "stats_path": stats_path,
        }
    return by_key, written_paths


def build_models(search_root: Path, output_dir: Path) -> tuple[list[dict], list[Path]]:
    config_dir = output_dir / "defense_configs"
    sparse_by_key, written_configs = build_sparse_config_lookup(search_root, config_dir)
    models: list[dict] = []
    for base_model in BASE_MODELS:
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
        for variant in SPARSE_VARIANTS:
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


def build_cases(models: list[dict]) -> list[dict]:
    cases: list[dict] = []
    baseline_sources = [item for item in models if item["variant"] == "baseline"]
    grouped_targets: dict[tuple[str, str], list[dict]] = {}
    for item in models:
        grouped_targets.setdefault((item["display_name"], item["regime"]), []).append(item)
    for group in grouped_targets.values():
        group.sort(key=lambda row: {"baseline": 0, "meansparse": 1, "extrasparse": 2}[row["variant"]])

    for attack in TRANSFER_ATTACKS:
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

    models, written_configs = build_models(search_root, output_dir)
    cases = build_cases(models)
    manifest = {
        "search_root": str(search_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_models": len(models),
        "num_cases": len(cases),
        "transfer_attacks": TRANSFER_ATTACKS,
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
