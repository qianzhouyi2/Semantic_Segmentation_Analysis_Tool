from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from src.models.layers import DropPath


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


def resize_pos_embed(
    posemb: torch.Tensor,
    grid_old_shape: tuple[int, int] | None,
    grid_new_shape: tuple[int, int],
    num_extra_tokens: int,
) -> torch.Tensor:
    posemb_tok, posemb_grid = posemb[:, :num_extra_tokens], posemb[0, num_extra_tokens:]
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    return torch.cat([posemb_tok, posemb_grid], dim=1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self) -> "Attention":
        return self

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        del mask
        batch_size, num_tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, num_tokens, 3, self.heads, channels // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float, out_dim: int | None = None) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim or dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self) -> "FeedForward":
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: int,
        embed_dim: int,
        channels: int,
    ) -> None:
        super().__init__()
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch size.")
        self.image_size = image_size
        self.grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: int,
        n_layers: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_cls: int,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        distilled: bool = False,
        channels: int = 3,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, d_model, channels)
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.distilled = distilled

        extra_tokens = 2 if self.distilled else 1
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.head_dist = nn.Linear(d_model, n_cls)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + extra_tokens, d_model))

        dpr = [rate.item() for rate in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[layer_idx]) for layer_idx in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)
        self.pre_logits = nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=0.02)

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"pos_embed", "cls_token", "dist_token"}

    def forward(self, x: torch.Tensor, pre_neck: bool = False) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        patch_size = self.patch_size
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + int(self.distilled)
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (height // patch_size, width // patch_size),
                num_extra_tokens,
            )
        x = self.dropout(x + pos_embed)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if pre_neck:
            return x

        if self.distilled:
            x_token, x_dist = x[:, 0], x[:, 1]
            return (self.head(x_token) + self.head_dist(x_dist)) / 2

        return self.head(x[:, 0])

    def get_attention_map(self, x: torch.Tensor, layer_id: int) -> torch.Tensor:
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(f"Invalid layer_id={layer_id}; expected 0 <= layer_id < {self.n_layers}.")

        batch_size, _, height, width = x.shape
        patch_size = self.patch_size
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + int(self.distilled)
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (height // patch_size, width // patch_size),
                num_extra_tokens,
            )
        x = x + pos_embed

        for block_idx, block in enumerate(self.blocks):
            if block_idx < layer_id:
                x = block(x)
            else:
                return block(x, return_attention=True)

        raise RuntimeError("Unreachable code reached while computing attention map.")
