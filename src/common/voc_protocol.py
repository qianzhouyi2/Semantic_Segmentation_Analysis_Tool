from __future__ import annotations


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

VOC_TRANSFER_ATTACKS = (
    {"stem": "mi_fgsm", "name": "mi-fgsm", "config": "configs/attacks/mi_fgsm.yaml"},
    {"stem": "ni_di_ti", "name": "ni+di+ti", "config": "configs/attacks/ni_di_ti.yaml"},
)


def checkpoint_name_to_voc_model(name: str) -> dict:
    for item in VOC_BASE_MODELS:
        if item["name"] == name:
            return dict(item)
    raise KeyError(f"Unknown VOC base model name: {name}")
