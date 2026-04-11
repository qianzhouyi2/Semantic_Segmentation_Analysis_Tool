from __future__ import annotations

from collections import OrderedDict
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import VisionTransformer
from src.models.heads import DecoderLinear, MaskTransformer


SEGMENTER_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
SEGMENTER_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)


def pad_to_patch_size(x: torch.Tensor, patch_size: int, fill_value: float = 0.0) -> torch.Tensor:
    height, width = x.size(2), x.size(3)
    pad_h = (patch_size - (height % patch_size)) % patch_size
    pad_w = (patch_size - (width % patch_size)) % patch_size
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), value=fill_value)


def remove_padding(x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    height, width = target_size
    return x[:, :, :height, :width]


class SegMenter(nn.Module):
    def __init__(self, encoder: VisionTransformer, decoder: nn.Module, n_cls: int, backbone: str) -> None:
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = 16
        self.encoder = encoder
        self.decoder = decoder
        self.backbone = backbone

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        def with_prefix(prefix: str, module: nn.Module) -> set[str]:
            if hasattr(module, "no_weight_decay"):
                return {prefix + name for name in module.no_weight_decay()}
            return set()

        return with_prefix("encoder.", self.encoder).union(with_prefix("decoder.", self.decoder))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = (x.size(2), x.size(3))
        x = pad_to_patch_size(x, self.patch_size)
        padded_size = (x.size(2), x.size(3))
        x = self.encoder(x, pre_neck=True)

        num_extra_tokens = 0 if "SAM" in self.backbone else 1 + int(self.encoder.distilled)
        x = x[:, num_extra_tokens:]
        masks = self.decoder(x, padded_size)
        masks = F.interpolate(masks, size=padded_size, mode="bilinear")
        return remove_padding(masks, original_size)

    def get_attention_map_enc(self, x: torch.Tensor, layer_id: int) -> torch.Tensor:
        return self.encoder.get_attention_map(x, layer_id)

    def get_attention_map_dec(self, x: torch.Tensor, layer_id: int) -> torch.Tensor:
        x = self.encoder(x, pre_neck=True)
        num_extra_tokens = 1 + int(self.encoder.distilled)
        x = x[:, num_extra_tokens:]
        return self.decoder.get_attention_map(x, layer_id)


class ImageNormalizer(nn.Module):
    def __init__(self, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def normalize_model(
    model: nn.Module,
    mean: tuple[float, float, float] = SEGMENTER_MEAN,
    std: tuple[float, float, float] = SEGMENTER_STD,
) -> nn.Module:
    return nn.Sequential(OrderedDict([("normalize", ImageNormalizer(mean, std)), ("model", model)]))


def build_segmenter_vit_small_config(num_classes: int = 21, image_size: int = 512) -> dict[str, object]:
    return {
        "image_size": (image_size, image_size),
        "patch_size": 16,
        "n_layers": 12,
        "d_model": 384,
        "n_heads": 6,
        "normalization": "vit",
        "distilled": False,
        "backbone": "vit_small_patch16_224",
        "dropout": 0.0,
        "drop_path_rate": 0.1,
        "decoder": {
            "name": "mask_transformer",
            "drop_path_rate": 0.0,
            "dropout": 0.1,
            "n_layers": 2,
        },
        "n_cls": num_classes,
    }


def create_vit(model_cfg: dict[str, object]) -> VisionTransformer:
    cfg = dict(model_cfg)
    cfg.pop("backbone", None)
    cfg.pop("normalization", None)
    cfg["n_cls"] = 1000
    cfg["d_ff"] = 4 * int(cfg["d_model"])
    return VisionTransformer(**cfg)


def create_decoder(encoder: VisionTransformer, decoder_cfg: dict[str, object], backbone: str) -> nn.Module:
    cfg = dict(decoder_cfg)
    name = str(cfg.pop("name"))
    cfg["d_encoder"] = 768 if "SAM" in backbone else 384
    cfg["patch_size"] = 16

    if "linear" in name:
        return DecoderLinear(**cfg)
    if name == "mask_transformer":
        dim = int(cfg["d_encoder"])
        cfg["n_heads"] = dim // 64
        cfg["d_model"] = dim
        cfg["d_ff"] = 4 * dim
        return MaskTransformer(**cfg)
    raise ValueError(f"Unknown Segmenter decoder: {name}")


def create_segmenter(model_cfg: dict[str, object], backbone: str) -> SegMenter:
    cfg = dict(model_cfg)
    decoder_cfg = dict(cfg.pop("decoder"))
    decoder_cfg["n_cls"] = cfg["n_cls"]
    encoder = create_vit(cfg)
    decoder = create_decoder(encoder, decoder_cfg, backbone=backbone)
    return SegMenter(encoder, decoder, n_cls=int(cfg["n_cls"]), backbone=backbone)


def build_segmenter_vit_small_patch16_224(num_classes: int = 21) -> SegMenter:
    cfg = build_segmenter_vit_small_config(num_classes=num_classes)
    return create_segmenter(cfg, backbone="vit_small_patch16_224")
