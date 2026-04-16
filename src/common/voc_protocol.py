from __future__ import annotations

from pathlib import Path

from src.common.config import load_yaml, resolve_project_path


VOC_BASE_MODELS = [
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

VOC_ATTACK_SUITE_TRANSFER_SOURCE_BY_FAMILY = {
    "segmenter_vit_s": "baseline__UperNet_ConvNext_T_VOC_clean",
    "upernet_convnext": "baseline__UperNet_ResNet50_VOC_clean",
    "upernet_resnet50": "baseline__Segmenter_ViT_S_VOC_clean",
}

VOC_DEFAULT_TRANSFER_ATTACK_STEMS = ("mi_fgsm", "ni_di_ti")

VOC_TRANSFER_ATTACK_LIBRARY = {
    "mi_fgsm": {"stem": "mi_fgsm", "name": "mi-fgsm", "config": "configs/attacks/mi_fgsm.yaml"},
    "ni_di_ti": {"stem": "ni_di_ti", "name": "ni+di+ti", "config": "configs/attacks/ni_di_ti.yaml"},
    "transegpgd": {"stem": "transegpgd", "name": "transegpgd", "config": "configs/attacks/transegpgd.yaml"},
    "tass": {"stem": "tass", "name": "tass", "config": "configs/attacks/tass.yaml"},
}

VOC_TRANSFER_ATTACKS = tuple(
    dict(VOC_TRANSFER_ATTACK_LIBRARY[stem]) for stem in VOC_DEFAULT_TRANSFER_ATTACK_STEMS
)


def resolve_transfer_attacks(
    attack_stems: list[str] | tuple[str, ...],
    *,
    config_dir: str | Path = "configs/attacks",
) -> list[dict]:
    if not attack_stems:
        raise ValueError("At least one transfer attack stem is required.")

    config_root = Path(config_dir)
    attacks: list[dict] = []
    seen: set[str] = set()
    for stem in attack_stems:
        if stem in seen:
            continue
        seen.add(stem)
        known_attack = VOC_TRANSFER_ATTACK_LIBRARY.get(stem)
        if known_attack is not None:
            attacks.append(dict(known_attack))
            continue

        relative_config_path = config_root / f"{stem}.yaml"
        config_path = resolve_project_path(relative_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Transfer attack config not found for stem `{stem}`: {config_path}")
        config_payload = load_yaml(config_path)
        attacks.append(
            {
                "stem": stem,
                "name": str(config_payload.get("name", stem)),
                "config": str(relative_config_path),
            }
        )
    return attacks


def checkpoint_name_to_voc_model(name: str) -> dict:
    for item in VOC_BASE_MODELS:
        if item["name"] == name:
            return dict(item)
    raise KeyError(f"Unknown VOC base model name: {name}")
