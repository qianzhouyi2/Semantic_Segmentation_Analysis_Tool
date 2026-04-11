from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tv_models

from src.models.backbones import CONVNEXT_SETTINGS, ConvNeXt


class UperNetConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        padding: int | tuple[int, int] | str = 0,
        bias: bool = False,
        dilation: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.activation(x)


class UperNetPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        self.layers = [
            nn.AdaptiveAvgPool2d(pool_scale),
            UperNetConvModule(in_channels, channels, kernel_size=1),
        ]
        for layer_idx, layer in enumerate(self.layers):
            self.add_module(str(layer_idx), layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class UperNetPyramidPoolingModule(nn.Module):
    def __init__(
        self,
        pool_scales: tuple[int, ...],
        in_channels: int,
        channels: int,
        align_corners: bool,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.blocks: list[UperNetPyramidPoolingBlock] = []
        for block_idx, pool_scale in enumerate(pool_scales):
            block = UperNetPyramidPoolingBlock(pool_scale, in_channels=in_channels, channels=channels)
            self.blocks.append(block)
            self.add_module(str(block_idx), block)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for block in self.blocks:
            pooled = block(x)
            outputs.append(
                nn.functional.interpolate(
                    pooled,
                    size=x.size()[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )
        return outputs


class UperNetHead(nn.Module):
    def __init__(self, in_channels: list[int] | tuple[int, ...], cls: int) -> None:
        super().__init__()
        self.pool_scales = (1, 2, 3, 6)
        self.in_channels = list(in_channels)
        self.channels = 512
        self.align_corners = False
        self.cls = cls
        self.classifier = nn.Conv2d(self.channels, self.cls, kernel_size=1)

        self.psp_modules = UperNetPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = UperNetConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        self.lateral_convs = nn.ModuleList(
            [UperNetConvModule(channels, self.channels, kernel_size=1) for channels in self.in_channels[:-1]]
        )
        self.fpn_convs = nn.ModuleList(
            [UperNetConvModule(self.channels, self.channels, kernel_size=3, padding=1) for _ in self.in_channels[:-1]]
        )
        self.fpn_bottleneck = UperNetConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def psp_forward(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x = inputs[-1]
        psp_outs = [x, *self.psp_modules(x)]
        return self.bottleneck(torch.cat(psp_outs, dim=1))

    def forward(self, encoder_hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        laterals = [
            lateral_conv(encoder_hidden_states[idx]) for idx, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(encoder_hidden_states))

        used_backbone_levels = len(laterals)
        for level_idx in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[level_idx - 1].shape[2:]
            laterals[level_idx - 1] = laterals[level_idx - 1] + nn.functional.interpolate(
                laterals[level_idx],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        fpn_outs = [self.fpn_convs[idx](laterals[idx]) for idx in range(used_backbone_levels - 1)]
        fpn_outs.append(laterals[-1])

        for level_idx in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[level_idx] = nn.functional.interpolate(
                fpn_outs[level_idx],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        x = torch.cat(fpn_outs, dim=1)
        x = self.fpn_bottleneck(x)
        return self.classifier(x)


class UperNetFCNHead(nn.Module):
    def __init__(
        self,
        in_index: int = 2,
        kernel_size: int = 3,
        dilation: int = 1,
        in_channels: int = 384,
        cls: int = 150,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = 256
        self.num_convs = 1
        self.concat_input = False
        self.in_index = in_index
        self.cls = cls

        conv_padding = (kernel_size // 2) * dilation
        convs = [
            UperNetConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
            )
        ]
        for _ in range(self.num_convs - 1):
            convs.append(
                UperNetConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                )
            )
        self.convs = nn.Sequential(*convs) if self.num_convs > 0 else nn.Identity()
        if self.concat_input:
            self.conv_cat = UperNetConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        self.classifier = nn.Conv2d(self.channels, self.cls, kernel_size=1)

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, encoder_hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        hidden_states = encoder_hidden_states[self.in_index]
        x = self.convs(hidden_states)
        if self.concat_input:
            x = self.conv_cat(torch.cat([hidden_states, x], dim=1))
        return self.classifier(x)


class TorchvisionResNetBackbone(nn.Module):
    def __init__(self, resnet: nn.Module) -> None:
        super().__init__()
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


class UperNetForSemanticSegmentation(nn.Module):
    def __init__(self, backbone: str = "ConvNeXt-T", n_cls: int = 150, pretrained: str | None = None) -> None:
        super().__init__()

        backbone_name = str(backbone)
        backbone_name_lower = backbone_name.lower()
        if backbone_name_lower.startswith("convnext-"):
            arch, variant = "ConvNeXt", backbone_name.split("-", 1)[1]
        elif backbone_name_lower.startswith("resnet-"):
            arch, variant = "ResNet", backbone_name.split("-", 1)[1]
        elif backbone_name_lower.startswith("resnet"):
            arch, variant = "ResNet", backbone_name[len("ResNet") :]
        else:
            raise ValueError(f"Unsupported backbone format: {backbone_name}")

        arch_lower = arch.lower()
        if arch_lower == "convnext":
            self.backbone = ConvNeXt(variant)
            in_channels = CONVNEXT_SETTINGS[variant][1]
            aux_in_channels = CONVNEXT_SETTINGS[variant][-2]
            del pretrained
        elif arch_lower == "resnet":
            if str(variant) == "50":
                net = tv_models.resnet50(weights=None)
            elif str(variant) == "101":
                net = tv_models.resnet101(weights=None)
            else:
                raise ValueError(f"Unsupported ResNet variant: {variant}")
            self.backbone = TorchvisionResNetBackbone(net)
            in_channels = [256, 512, 1024, 2048]
            aux_in_channels = in_channels[-2]
            del pretrained
        else:
            raise ValueError(f"Unsupported backbone arch: {arch}")

        self.decode_head = UperNetHead(in_channels=in_channels, cls=n_cls)
        self.auxiliary_head = UperNetFCNHead(in_channels=aux_in_channels, cls=n_cls)
        self.decode_head.init_weights()
        self.auxiliary_head.init_weights()

    def forward(self, input: torch.Tensor | None = None, lbl: torch.Tensor | None = None):
        features = self.backbone(input)
        logits = self.decode_head(features)
        logits = nn.functional.interpolate(logits, size=input.shape[2:], mode="bilinear", align_corners=False)

        loss = None
        if lbl is not None:
            auxiliary_logits = self.auxiliary_head(features)
            auxiliary_logits = nn.functional.interpolate(
                auxiliary_logits,
                size=input.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            auxiliary_loss = loss_fct(auxiliary_logits, lbl)
            main_loss = loss_fct(logits, lbl)
            loss = main_loss + 0.4 * auxiliary_loss

        if self.training:
            return loss, logits
        return logits
