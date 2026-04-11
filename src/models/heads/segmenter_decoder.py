from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from src.models.backbones.vit import Block, init_weights


class DecoderLinear(nn.Module):
    def __init__(self, n_cls: int, patch_size: int, d_encoder: int) -> None:
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls
        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return set()

    def forward(self, x: torch.Tensor, im_size: tuple[int, int]) -> torch.Tensor:
        height, _ = im_size
        grid_size = height // self.patch_size
        x = self.head(x)
        batch_size, _, channels = x.shape
        return x.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)


class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls: int,
        patch_size: int,
        d_encoder: int,
        n_layers: int,
        n_heads: int,
        d_model: int,
        d_ff: int,
        drop_path_rate: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model**-0.5

        dpr = [rate.item() for rate in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[layer_idx]) for layer_idx in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)
        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"cls_emb"}

    def forward(self, x: torch.Tensor, im_size: tuple[int, int]) -> torch.Tensor:
        height, _ = im_size
        grid_size = height // self.patch_size
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, :-self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        batch_size = masks.shape[0]
        return masks.transpose(1, 2).reshape(batch_size, self.n_cls, int(grid_size), int(grid_size))

    def get_attention_map(self, x: torch.Tensor, layer_id: int) -> torch.Tensor:
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(f"Invalid layer_id={layer_id}; expected 0 <= layer_id < {self.n_layers}.")

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), dim=1)
        for block_idx, block in enumerate(self.blocks):
            if block_idx < layer_id:
                x = block(x)
            else:
                return block(x, return_attention=True)
        raise RuntimeError("Unreachable code reached while computing decoder attention map.")
