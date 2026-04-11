from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from src.models.layers import DropPath

class ConvStem(nn.Module):
    def __init__(self, channels: int = 48) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1),
            LayerNorm(channels, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(channels * 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


CONVNEXT_SETTINGS: dict[str, list[object]] = {
    "T": [[3, 3, 9, 3], [96, 192, 384, 768], 384, 0.4],
    "T_CVST": [[3, 3, 9, 3], [96, 192, 384, 768], 384, 0.4],
    "T_CVST_ROB": [[3, 3, 9, 3], [96, 192, 384, 768], 384, 0.4],
    "S_CVST_ROB": [[3, 3, 27, 3], [96, 192, 384, 768], 384, 0.3],
    "S_CVST": [[3, 3, 27, 3], [96, 192, 384, 768], 384, 0.3],
    "B": [[3, 3, 27, 3], [128, 256, 512, 1024], 512, 0.4],
}


class ConvNeXt(nn.Module):
    def __init__(
        self,
        variant: str,
        in_chans: int = 3,
        layer_scale_init_value: float = 1.0,
        out_indices: list[int] | tuple[int, ...] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        if variant not in CONVNEXT_SETTINGS:
            raise ValueError(
                f"ConvNeXt variant must be one of {sorted(CONVNEXT_SETTINGS)}, got {variant!r}."
            )

        depths, dims, _, drop_path_rate = CONVNEXT_SETTINGS[variant]
        self.variant = variant
        self.downsample_layers = nn.ModuleList()
        stem = (
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
            if "CVST" not in variant
            else ConvStem()
        )
        self.downsample_layers.append(stem)

        for stage_idx in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[stage_idx], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[stage_idx], dims[stage_idx + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        dp_rates = [rate.item() for rate in torch.linspace(0, drop_path_rate, sum(depths))]
        offset = 0
        for stage_idx in range(4):
            blocks = [
                ConvNeXtBlock(
                    dim=dims[stage_idx],
                    drop_path=dp_rates[offset + block_idx],
                    layer_scale_init_value=layer_scale_init_value,
                )
                for block_idx in range(depths[stage_idx])
            ]
            self.stages.append(nn.Sequential(*blocks))
            offset += depths[stage_idx]

        self.out_indices = tuple(out_indices)
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for stage_idx in range(4):
            self.add_module(f"norm{stage_idx}", norm_layer(dims[stage_idx]))

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = []
        for stage_idx in range(4):
            x = self.downsample_layers[stage_idx](x)
            x = self.stages[stage_idx](x)
            if stage_idx in self.out_indices:
                norm_layer = getattr(self, f"norm{stage_idx}")
                outputs.append(norm_layer(x))
        return tuple(outputs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.forward_features(x)


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in {"channels_last", "channels_first"}:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]
