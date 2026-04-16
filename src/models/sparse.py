from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import types
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

from src.models.architectures.segmenter import SegMenter
from src.models.architectures.upernet import TorchvisionResNetBackbone, UperNetForSemanticSegmentation
from src.models.backbones.convnext import ConvNeXt, ConvNeXtBlock, ConvStem
from src.models.backbones.vit import Block as ViTBlock
from src.models.backbones.vit import VisionTransformer
from src.models.heads import DecoderLinear, MaskTransformer


SPARSE_DEFENSE_CHOICES = (
    "meansparse",
    "extrasparse",
    "cc_extra_sparse",
    "dir_extra_sparse",
    "margin_extra_sparse",
)
POSTSPARSE_VARIANTS = {
    "cc_extra_sparse",
    "dir_extra_sparse",
    "margin_extra_sparse",
}
SUPPORTED_SPARSE_FAMILIES = {
    "upernet_convnext",
    "upernet_resnet50",
    "segmenter_vit_s",
}
SPARSE_SIDECAR_FORMAT_VERSION = 1


@dataclass(slots=True)
class SparseDefenseConfig:
    variant: str
    stats_path: Path | None = None
    threshold: float = 0.0
    direction_mode: str = "weight_sign"
    lambda_mix: float = 0.5
    alpha0: float | None = None
    alpha0_mode: str = "fixed"
    beta: float = 0.15
    beta_scale: float | None = None
    tau: float = 1.5
    strict_stats: bool = True
    family: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.variant not in SPARSE_DEFENSE_CHOICES:
            raise ValueError(
                f"Unknown sparse defense variant: {self.variant}. Expected one of {SPARSE_DEFENSE_CHOICES}."
            )
        self.threshold = float(self.threshold)
        self.lambda_mix = float(self.lambda_mix)
        self.beta = float(self.beta)
        self.tau = float(self.tau)
        if self.alpha0 is not None:
            self.alpha0 = float(self.alpha0)
        if self.beta_scale is not None:
            self.beta_scale = float(self.beta_scale)

    @property
    def is_postsparse(self) -> bool:
        return self.variant in POSTSPARSE_VARIANTS

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        base_dir: str | Path | None = None,
    ) -> SparseDefenseConfig:
        variant = payload.get("variant", payload.get("name"))
        if not variant:
            raise ValueError("Sparse defense config requires `variant` or `name`.")
        stats_path = payload.get("stats_path")
        resolved_stats_path: Path | None = None
        if stats_path:
            candidate = Path(stats_path)
            if not candidate.is_absolute() and base_dir is not None:
                candidate = Path(base_dir) / candidate
            resolved_stats_path = candidate
        return cls(
            variant=str(variant),
            stats_path=resolved_stats_path,
            threshold=float(payload.get("threshold", 0.0)),
            direction_mode=str(payload.get("direction_mode", "weight_sign")),
            lambda_mix=float(payload.get("lambda_mix", 0.5)),
            alpha0=None if payload.get("alpha0") is None else float(payload["alpha0"]),
            alpha0_mode=str(payload.get("alpha0_mode", "fixed")),
            beta=float(payload.get("beta", 0.15)),
            beta_scale=None if payload.get("beta_scale") is None else float(payload["beta_scale"]),
            tau=float(payload.get("tau", 1.5)),
            strict_stats=bool(payload.get("strict_stats", True)),
            family=None if payload.get("family") is None else str(payload["family"]),
            name=None if payload.get("name") is None else str(payload["name"]),
        )


def load_sparse_defense_config(path: str | Path) -> SparseDefenseConfig:
    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in sparse defense config: {config_path}")
    return SparseDefenseConfig.from_dict(payload, base_dir=config_path.parent)


def supports_sparse_defense(family: str) -> bool:
    return family in SUPPORTED_SPARSE_FAMILIES


class _IdentitySTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def _apply_identity_ste(x: torch.Tensor, output: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return output
    return _IdentitySTEFunction.apply(x, output)


class _RunningStatsSparse2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.zeros(channels))
        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("flag_update_statistics", torch.tensor(0))
        self.register_buffer("batch_num", torch.tensor(0.0))
        self.attack_backward_mode = "default"

    def reset_statistics(self) -> None:
        self.running_mean.zero_()
        self.running_var.zero_()
        self.flag_update_statistics.zero_()
        self.batch_num.zero_()

    def start_statistics(self, num_batches: int) -> None:
        self.reset_statistics()
        self.flag_update_statistics.fill_(1)
        self.batch_num.fill_(float(num_batches))

    def stop_statistics(self) -> None:
        self.flag_update_statistics.zero_()

    def set_threshold(self, threshold: float) -> None:
        self.threshold.fill_(float(threshold))

    def _update_statistics(self, x: torch.Tensor) -> None:
        if bool(self.flag_update_statistics.item()):
            self.running_mean += torch.mean(x.detach(), dim=(0, 2, 3)) / self.batch_num
            self.running_var += torch.var(x.detach(), dim=(0, 2, 3)) / self.batch_num

    def _bias_and_crop(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        threshold = float(self.threshold.item())
        bias = self.running_mean.view(1, -1, 1, 1)
        crop = threshold * torch.sqrt(torch.clamp(self.running_var, min=1e-12)).view(1, -1, 1, 1)
        return bias, crop, threshold

    def use_bpda_ste(self) -> bool:
        return self.attack_backward_mode == "bpda_ste"


class MeanSparse2d(_RunningStatsSparse2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._update_statistics(x)
        bias, crop, threshold = self._bias_and_crop()
        if threshold == 0.0:
            return x
        diff = x - bias
        output = torch.where(torch.abs(diff) < crop, bias.expand_as(x), x)
        return _apply_identity_ste(x, output, self.use_bpda_ste())


class ExtraSparse2d(_RunningStatsSparse2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._update_statistics(x)
        bias, crop, threshold = self._bias_and_crop()
        if threshold == 0.0:
            return x
        diff = x - bias
        inside = torch.abs(diff) < crop
        upper = bias + crop
        lower = bias - crop
        target = torch.where(diff >= 0, upper, lower)
        output = torch.where(inside, target, x)
        return _apply_identity_ste(x, output, self.use_bpda_ste())


class PostSparse2d(_RunningStatsSparse2d):
    def __init__(
        self,
        channels: int,
        *,
        num_classes: int = 21,
        mode: str,
        eps: float = 1e-6,
        direction_mode: str = "weight_sign",
        lambda_mix: float = 0.5,
        alpha0: float | None = None,
        alpha0_mode: str = "fixed",
        beta: float = 0.15,
        beta_scale: float | None = None,
        tau: float = 1.5,
    ) -> None:
        super().__init__(channels)
        self.num_classes = int(num_classes)
        self.mode = mode
        self.eps = float(eps)
        self.direction_mode = direction_mode
        self.lambda_mix = float(lambda_mix)
        self.alpha0 = alpha0
        self.alpha0_mode = alpha0_mode
        self.beta = float(beta)
        self.beta_scale = beta_scale
        self.tau = float(tau)
        self.threshold_override: torch.Tensor | float | None = None
        self.soft_change_temperature = 0.05
        self.last_soft_changed_fraction: torch.Tensor | None = None
        self.register_buffer("class_conditional_mean", torch.zeros(self.num_classes, channels))
        self.register_buffer("class_conditional_std", torch.ones(self.num_classes, channels))
        self.register_buffer("class_count", torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer("classifier_weight", torch.zeros(self.num_classes, channels))
        self.runtime_enabled = True
        self.runtime_pred: torch.Tensor | None = None
        self.runtime_logits: torch.Tensor | None = None
        self.runtime_margin: torch.Tensor | None = None

    def reset_statistics(self) -> None:
        super().reset_statistics()
        self.class_conditional_mean.zero_()
        self.class_conditional_std.fill_(1.0)
        self.class_count.zero_()
        self.classifier_weight.zero_()
        self.clear_runtime_context()

    def set_class_statistics(
        self,
        mean: torch.Tensor | None,
        std: torch.Tensor | None,
        count: torch.Tensor | None = None,
    ) -> None:
        if mean is None or std is None:
            self.class_conditional_mean.zero_()
            self.class_conditional_std.fill_(1.0)
            self.class_count.zero_()
            return
        mean = mean.to(device=self.class_conditional_mean.device, dtype=self.class_conditional_mean.dtype)
        std = std.to(device=self.class_conditional_std.device, dtype=self.class_conditional_std.dtype)
        if mean.shape != self.class_conditional_mean.shape or std.shape != self.class_conditional_std.shape:
            raise ValueError(
                f"Invalid class statistics shape for {self.mode}: "
                f"mean={tuple(mean.shape)} std={tuple(std.shape)} "
                f"expected={tuple(self.class_conditional_mean.shape)}"
            )
        self.class_conditional_mean.copy_(mean)
        self.class_conditional_std.copy_(torch.clamp(std, min=self.eps))
        if count is None:
            self.class_count.zero_()
        else:
            count = count.to(device=self.class_count.device, dtype=self.class_count.dtype)
            if count.shape != self.class_count.shape:
                raise ValueError(
                    f"Invalid class-count shape for {self.mode}: {tuple(count.shape)} != {tuple(self.class_count.shape)}"
                )
            self.class_count.copy_(count)

    def set_classifier_weight(self, weight: torch.Tensor | None) -> None:
        self.classifier_weight.zero_()
        if weight is None:
            return
        weight = weight.detach()
        if weight.ndim == 4 and weight.shape[2:] == (1, 1):
            weight = weight[:, :, 0, 0]
        if weight.ndim != 2 or weight.shape != self.classifier_weight.shape:
            return
        weight = weight.to(device=self.classifier_weight.device, dtype=self.classifier_weight.dtype)
        self.classifier_weight.copy_(weight)

    def set_runtime_context(
        self,
        *,
        pred: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        margin: torch.Tensor | None = None,
    ) -> None:
        self.runtime_pred = pred
        self.runtime_logits = logits
        self.runtime_margin = margin

    def clear_runtime_context(self) -> None:
        self.runtime_pred = None
        self.runtime_logits = None
        self.runtime_margin = None

    def _resolve_threshold(self) -> torch.Tensor | float:
        return self.threshold_override if self.threshold_override is not None else self.threshold

    def _global_bias_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        bias = self.running_mean.view(1, self.running_mean.shape[0], 1, 1)
        std = torch.sqrt(torch.clamp(self.running_var, min=self.eps)).view(1, self.running_var.shape[0], 1, 1)
        return bias, std

    def _resize_pred(self, pred: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
        if pred is None:
            return None
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred[:, 0]
        if pred.ndim == 1:
            return pred.to(device=x.device, dtype=torch.long)
        if pred.ndim != 3:
            return None
        pred = pred.to(device=x.device, dtype=torch.long)
        if pred.shape[-2:] != x.shape[-2:]:
            pred = F.interpolate(pred.unsqueeze(1).float(), size=x.shape[-2:], mode="nearest").squeeze(1).long()
        return pred

    def _resize_logits(self, logits: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
        if logits is None or logits.ndim != 4:
            return None
        logits = logits.to(device=x.device, dtype=x.dtype)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def _resolve_pred(self, x: torch.Tensor) -> torch.Tensor | None:
        pred = self._resize_pred(self.runtime_pred, x)
        if pred is not None:
            return pred
        logits = self._resize_logits(self.runtime_logits, x)
        if logits is None:
            return None
        return logits.argmax(dim=1)

    def _resolve_margin(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.runtime_margin is not None:
            margin = self.runtime_margin
            if margin.ndim == 4 and margin.shape[1] == 1:
                margin = margin[:, 0]
            if margin.ndim == 1:
                return margin.to(device=x.device, dtype=x.dtype)
            if margin.ndim == 3:
                margin = margin.to(device=x.device, dtype=x.dtype)
                if margin.shape[-2:] != x.shape[-2:]:
                    margin = F.interpolate(
                        margin.unsqueeze(1),
                        size=x.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                return margin
        logits = self._resize_logits(self.runtime_logits, x)
        if logits is None or logits.shape[1] < 2:
            return None
        top2 = logits.topk(k=2, dim=1).values
        return top2[:, 0] - top2[:, 1]

    def _expand_alpha(self, alpha: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(alpha):
            alpha = alpha.to(device=x.device, dtype=x.dtype)
            if alpha.ndim == 0:
                return alpha.view(1, 1, 1, 1)
            if alpha.ndim == 1:
                if alpha.shape[0] == x.shape[0]:
                    return alpha.view(-1, 1, 1, 1)
                if alpha.shape[0] == x.shape[1]:
                    return alpha.view(1, -1, 1, 1)
            if alpha.ndim == 3:
                return alpha.unsqueeze(1)
            return alpha
        return torch.tensor(alpha, device=x.device, dtype=x.dtype).view(1, 1, 1, 1)

    def _safe_sign(self, value: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        sign = torch.sign(value)
        return torch.where(sign == 0, fallback, sign)

    def _raw_direction(self, diff: torch.Tensor) -> torch.Tensor:
        return torch.where(diff >= 0, torch.ones_like(diff), -torch.ones_like(diff))

    def _select_class_stats(
        self,
        pred: torch.Tensor | None,
        x: torch.Tensor,
        fallback_bias: torch.Tensor,
        fallback_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pred is None or int(self.class_count.sum().item()) == 0:
            return fallback_bias, fallback_std
        pred = pred.clamp(0, self.num_classes - 1)
        if pred.ndim == 1:
            class_mean = self.class_conditional_mean.index_select(0, pred).view(x.shape[0], x.shape[1], 1, 1)
            class_std = self.class_conditional_std.index_select(0, pred).view(x.shape[0], x.shape[1], 1, 1)
            valid = self.class_count.index_select(0, pred).view(-1, 1, 1, 1) > 0
        else:
            flat_pred = pred.reshape(-1)
            class_mean = self.class_conditional_mean.index_select(0, flat_pred)
            class_std = self.class_conditional_std.index_select(0, flat_pred)
            class_mean = class_mean.view(x.shape[0], x.shape[2], x.shape[3], x.shape[1]).permute(0, 3, 1, 2)
            class_std = class_std.view(x.shape[0], x.shape[2], x.shape[3], x.shape[1]).permute(0, 3, 1, 2)
            valid = self.class_count.index_select(0, flat_pred).view(x.shape[0], 1, x.shape[2], x.shape[3]) > 0
        class_mean = class_mean.to(device=x.device, dtype=x.dtype)
        class_std = torch.clamp(class_std.to(device=x.device, dtype=x.dtype), min=self.eps)
        return (
            torch.where(valid, class_mean, fallback_bias),
            torch.where(valid, class_std, fallback_std),
        )

    def _resolve_direction(
        self,
        diff: torch.Tensor,
        std: torch.Tensor,
        pred: torch.Tensor | None,
    ) -> torch.Tensor:
        raw_direction = self._raw_direction(diff)
        if self.direction_mode == "raw_sign":
            return raw_direction
        if pred is None or self.classifier_weight.abs().sum().item() == 0:
            return raw_direction
        if pred.ndim == 1:
            class_weight = self.classifier_weight.index_select(0, pred).view(diff.shape[0], diff.shape[1], 1, 1)
        else:
            flat_pred = pred.reshape(-1)
            class_weight = self.classifier_weight.index_select(0, flat_pred)
            class_weight = class_weight.view(diff.shape[0], diff.shape[2], diff.shape[3], diff.shape[1]).permute(
                0, 3, 1, 2
            )
        class_weight = class_weight.to(device=diff.device, dtype=diff.dtype)
        weight_direction = self._safe_sign(class_weight, raw_direction)
        if self.direction_mode == "weight_sign":
            return weight_direction
        if self.direction_mode == "mixed":
            normalized_diff = diff / (std + self.eps)
            mixed_score = self.lambda_mix * normalized_diff + (1.0 - self.lambda_mix) * class_weight
            return self._safe_sign(mixed_score, raw_direction)
        raise ValueError(f"Unsupported direction mode: {self.direction_mode}")

    def _resolve_base_alpha(self, threshold: torch.Tensor | float) -> torch.Tensor | float:
        if self.alpha0_mode == "threshold":
            return threshold
        return threshold if self.alpha0 is None else self.alpha0

    def _resolve_beta(self, threshold: torch.Tensor | float) -> torch.Tensor | float:
        if self.beta_scale is None:
            return self.beta
        if torch.is_tensor(threshold):
            return threshold * float(self.beta_scale)
        return float(threshold) * float(self.beta_scale)

    def _resolve_alpha(self, threshold: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        alpha = self._expand_alpha(self._resolve_base_alpha(threshold), x)
        if self.mode != "margin_extra_sparse":
            return alpha
        margin = self._resolve_margin(x)
        if margin is None:
            return alpha
        margin = margin.to(device=x.device, dtype=x.dtype)
        tau = max(self.tau, self.eps)
        adaptive = self._expand_alpha(self._resolve_beta(threshold), x) * torch.exp(-margin / tau).unsqueeze(1)
        return alpha + adaptive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._update_statistics(x)
        if not self.runtime_enabled:
            return x
        threshold = self._resolve_threshold()
        bias, std = self._global_bias_std()
        pred = self._resolve_pred(x)
        if self.mode == "cc_extra_sparse":
            bias, std = self._select_class_stats(pred, x, bias, std)
        crop = self._resolve_alpha(threshold, x) * std
        diff = x - bias
        if self.mode == "cc_extra_sparse":
            direction = self._raw_direction(diff)
        elif self.mode in {"dir_extra_sparse", "margin_extra_sparse"}:
            direction = self._resolve_direction(diff, std, pred)
        else:
            raise ValueError(f"Unsupported PostSparse mode: {self.mode}")
        inside = torch.abs(diff) < crop
        self.last_soft_changed_fraction = torch.sigmoid((crop - torch.abs(diff)) / self.soft_change_temperature).mean()
        target = bias + direction * crop
        output = torch.where(inside, target, x)
        return _apply_identity_ste(x, output, self.use_bpda_ste())


class CCExtraSparse2d(PostSparse2d):
    def __init__(self, channels: int, *, num_classes: int = 21) -> None:
        super().__init__(channels, mode="cc_extra_sparse", num_classes=num_classes)


class DirExtraSparse2d(PostSparse2d):
    def __init__(self, channels: int, *, num_classes: int = 21) -> None:
        super().__init__(channels, mode="dir_extra_sparse", num_classes=num_classes)


class MarginExtraSparse2d(PostSparse2d):
    def __init__(self, channels: int, *, num_classes: int = 21) -> None:
        super().__init__(channels, mode="margin_extra_sparse", num_classes=num_classes)


class _RunningStatsSparseTokens(nn.Module):
    def __init__(self, channels: int, *, num_extra_tokens: int = 1) -> None:
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.zeros(channels))
        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("flag_update_statistics", torch.tensor(0))
        self.register_buffer("batch_num", torch.tensor(0.0))
        self.num_extra_tokens = int(num_extra_tokens)
        self.runtime_grid_size: tuple[int, int] | None = None
        self.attack_backward_mode = "default"

    def reset_statistics(self) -> None:
        self.running_mean.zero_()
        self.running_var.zero_()
        self.flag_update_statistics.zero_()
        self.batch_num.zero_()

    def start_statistics(self, num_batches: int) -> None:
        self.reset_statistics()
        self.flag_update_statistics.fill_(1)
        self.batch_num.fill_(float(num_batches))

    def stop_statistics(self) -> None:
        self.flag_update_statistics.zero_()

    def set_threshold(self, threshold: float) -> None:
        self.threshold.fill_(float(threshold))

    def set_grid_size(self, grid_size: tuple[int, int] | None) -> None:
        self.runtime_grid_size = None if grid_size is None else (int(grid_size[0]), int(grid_size[1]))

    def _infer_grid_size(self, x: torch.Tensor) -> tuple[int, int]:
        if self.runtime_grid_size is not None:
            return self.runtime_grid_size
        num_patch_tokens = x.shape[1] - self.num_extra_tokens
        side = int(round(num_patch_tokens**0.5))
        if side * side != num_patch_tokens:
            raise ValueError(f"Cannot infer token grid size from sequence length {x.shape[1]} with {self.num_extra_tokens} extra tokens.")
        return side, side

    def _update_statistics(self, x: torch.Tensor) -> None:
        if bool(self.flag_update_statistics.item()):
            self.running_mean += torch.mean(x.detach(), dim=(0, 1)) / self.batch_num
            self.running_var += torch.var(x.detach(), dim=(0, 1)) / self.batch_num

    def _bias_and_crop(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        threshold = float(self.threshold.item())
        bias = self.running_mean.view(1, 1, -1)
        crop = threshold * torch.sqrt(torch.clamp(self.running_var, min=1e-12)).view(1, 1, -1)
        return bias, crop, threshold

    def use_bpda_ste(self) -> bool:
        return self.attack_backward_mode == "bpda_ste"


class MeanSparseTokens(_RunningStatsSparseTokens):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._update_statistics(x)
        bias, crop, threshold = self._bias_and_crop()
        if threshold == 0.0:
            return x
        diff = x - bias
        output = torch.where(torch.abs(diff) < crop, bias.expand_as(x), x)
        return _apply_identity_ste(x, output, self.use_bpda_ste())


class ExtraSparseTokens(_RunningStatsSparseTokens):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._update_statistics(x)
        bias, crop, threshold = self._bias_and_crop()
        if threshold == 0.0:
            return x
        diff = x - bias
        inside = torch.abs(diff) < crop
        upper = bias + crop
        lower = bias - crop
        target = torch.where(diff >= 0, upper, lower)
        output = torch.where(inside, target, x)
        return _apply_identity_ste(x, output, self.use_bpda_ste())


class PostSparseTokens(_RunningStatsSparseTokens):
    def __init__(
        self,
        channels: int,
        *,
        num_classes: int = 21,
        mode: str,
        num_extra_tokens: int = 1,
        eps: float = 1e-6,
        direction_mode: str = "weight_sign",
        lambda_mix: float = 0.5,
        alpha0: float | None = None,
        alpha0_mode: str = "fixed",
        beta: float = 0.15,
        beta_scale: float | None = None,
        tau: float = 1.5,
    ) -> None:
        super().__init__(channels, num_extra_tokens=num_extra_tokens)
        self.num_classes = int(num_classes)
        self.mode = mode
        self.eps = float(eps)
        self.direction_mode = direction_mode
        self.lambda_mix = float(lambda_mix)
        self.alpha0 = alpha0
        self.alpha0_mode = alpha0_mode
        self.beta = float(beta)
        self.beta_scale = beta_scale
        self.tau = float(tau)
        self.threshold_override: torch.Tensor | float | None = None
        self.soft_change_temperature = 0.05
        self.last_soft_changed_fraction: torch.Tensor | None = None
        self.register_buffer("class_conditional_mean", torch.zeros(self.num_classes, channels))
        self.register_buffer("class_conditional_std", torch.ones(self.num_classes, channels))
        self.register_buffer("class_count", torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer("classifier_weight", torch.zeros(self.num_classes, channels))
        self.runtime_enabled = True
        self.runtime_pred: torch.Tensor | None = None
        self.runtime_logits: torch.Tensor | None = None
        self.runtime_margin: torch.Tensor | None = None

    def reset_statistics(self) -> None:
        super().reset_statistics()
        self.class_conditional_mean.zero_()
        self.class_conditional_std.fill_(1.0)
        self.class_count.zero_()
        self.classifier_weight.zero_()
        self.clear_runtime_context()

    def set_class_statistics(
        self,
        mean: torch.Tensor | None,
        std: torch.Tensor | None,
        count: torch.Tensor | None = None,
    ) -> None:
        if mean is None or std is None:
            self.class_conditional_mean.zero_()
            self.class_conditional_std.fill_(1.0)
            self.class_count.zero_()
            return
        mean = mean.to(device=self.class_conditional_mean.device, dtype=self.class_conditional_mean.dtype)
        std = std.to(device=self.class_conditional_std.device, dtype=self.class_conditional_std.dtype)
        if mean.shape != self.class_conditional_mean.shape or std.shape != self.class_conditional_std.shape:
            raise ValueError(
                f"Invalid class statistics shape for {self.mode}: "
                f"mean={tuple(mean.shape)} std={tuple(std.shape)} "
                f"expected={tuple(self.class_conditional_mean.shape)}"
            )
        self.class_conditional_mean.copy_(mean)
        self.class_conditional_std.copy_(torch.clamp(std, min=self.eps))
        if count is None:
            self.class_count.zero_()
        else:
            count = count.to(device=self.class_count.device, dtype=self.class_count.dtype)
            if count.shape != self.class_count.shape:
                raise ValueError(
                    f"Invalid class-count shape for {self.mode}: {tuple(count.shape)} != {tuple(self.class_count.shape)}"
                )
            self.class_count.copy_(count)

    def set_classifier_weight(self, weight: torch.Tensor | None) -> None:
        self.classifier_weight.zero_()
        if weight is None:
            return
        if weight.ndim != 2 or weight.shape != self.classifier_weight.shape:
            return
        weight = weight.to(device=self.classifier_weight.device, dtype=self.classifier_weight.dtype)
        self.classifier_weight.copy_(weight)

    def set_runtime_context(
        self,
        *,
        pred: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        margin: torch.Tensor | None = None,
    ) -> None:
        self.runtime_pred = pred
        self.runtime_logits = logits
        self.runtime_margin = margin

    def clear_runtime_context(self) -> None:
        self.runtime_pred = None
        self.runtime_logits = None
        self.runtime_margin = None

    def _resolve_threshold(self) -> torch.Tensor | float:
        return self.threshold_override if self.threshold_override is not None else self.threshold

    def _global_bias_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        bias = self.running_mean.view(1, 1, self.running_mean.shape[0])
        std = torch.sqrt(torch.clamp(self.running_var, min=self.eps)).view(1, 1, self.running_var.shape[0])
        return bias, std

    def _resize_pred_grid(self, pred: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
        if pred is None:
            return None
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred[:, 0]
        if pred.ndim != 3:
            return None
        pred = pred.to(device=x.device, dtype=torch.long)
        grid_size = self._infer_grid_size(x)
        if pred.shape[-2:] != grid_size:
            pred = F.interpolate(pred.unsqueeze(1).float(), size=grid_size, mode="nearest").squeeze(1).long()
        return pred

    def _resize_logits_grid(self, logits: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
        if logits is None or logits.ndim != 4:
            return None
        logits = logits.to(device=x.device, dtype=x.dtype)
        grid_size = self._infer_grid_size(x)
        if logits.shape[-2:] != grid_size:
            logits = F.interpolate(logits, size=grid_size, mode="bilinear", align_corners=False)
        return logits

    def _resolve_pred_grid(self, x: torch.Tensor) -> torch.Tensor | None:
        pred = self._resize_pred_grid(self.runtime_pred, x)
        if pred is not None:
            return pred
        logits = self._resize_logits_grid(self.runtime_logits, x)
        if logits is None:
            return None
        return logits.argmax(dim=1)

    def _resolve_margin_grid(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.runtime_margin is not None:
            margin = self.runtime_margin
            if margin.ndim == 4 and margin.shape[1] == 1:
                margin = margin[:, 0]
            if margin.ndim == 3:
                margin = margin.to(device=x.device, dtype=x.dtype)
                grid_size = self._infer_grid_size(x)
                if margin.shape[-2:] != grid_size:
                    margin = F.interpolate(margin.unsqueeze(1), size=grid_size, mode="bilinear", align_corners=False)[:, 0]
                return margin
        logits = self._resize_logits_grid(self.runtime_logits, x)
        if logits is None or logits.shape[1] < 2:
            return None
        top2 = logits.topk(k=2, dim=1).values
        return top2[:, 0] - top2[:, 1]

    def _expand_alpha(self, alpha: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(alpha):
            alpha = alpha.to(device=x.device, dtype=x.dtype)
            if alpha.ndim == 0:
                return alpha.view(1, 1, 1)
            if alpha.ndim == 1:
                if alpha.shape[0] == x.shape[0]:
                    return alpha.view(-1, 1, 1)
                if alpha.shape[0] == x.shape[2]:
                    return alpha.view(1, 1, -1)
            if alpha.ndim == 2:
                return alpha.unsqueeze(-1)
            return alpha
        return torch.tensor(alpha, device=x.device, dtype=x.dtype).view(1, 1, 1)

    def _safe_sign(self, value: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        sign = torch.sign(value)
        return torch.where(sign == 0, fallback, sign)

    def _raw_direction(self, diff: torch.Tensor) -> torch.Tensor:
        return torch.where(diff >= 0, torch.ones_like(diff), -torch.ones_like(diff))

    def _patch_token_bias_std(
        self,
        patch_values: torch.Tensor,
        fallback_value: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_extra_tokens <= 0:
            return patch_values
        extra = fallback_value.expand(x.shape[0], self.num_extra_tokens, -1)
        return torch.cat((extra, patch_values), dim=1)

    def _select_class_stats(
        self,
        pred_grid: torch.Tensor | None,
        x: torch.Tensor,
        fallback_bias: torch.Tensor,
        fallback_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pred_grid is None or int(self.class_count.sum().item()) == 0:
            return fallback_bias, fallback_std
        grid_h, grid_w = self._infer_grid_size(x)
        flat_pred = pred_grid.clamp(0, self.num_classes - 1).reshape(-1)
        class_mean = self.class_conditional_mean.index_select(0, flat_pred).view(x.shape[0], grid_h * grid_w, x.shape[2])
        class_std = self.class_conditional_std.index_select(0, flat_pred).view(x.shape[0], grid_h * grid_w, x.shape[2])
        valid = self.class_count.index_select(0, flat_pred).view(x.shape[0], grid_h * grid_w, 1) > 0
        fallback_bias_patch = fallback_bias.expand(x.shape[0], grid_h * grid_w, -1)
        fallback_std_patch = fallback_std.expand(x.shape[0], grid_h * grid_w, -1)
        patch_bias = torch.where(valid, class_mean.to(device=x.device, dtype=x.dtype), fallback_bias_patch)
        patch_std = torch.where(
            valid,
            torch.clamp(class_std.to(device=x.device, dtype=x.dtype), min=self.eps),
            fallback_std_patch,
        )
        return (
            self._patch_token_bias_std(patch_bias, fallback_bias, x),
            self._patch_token_bias_std(patch_std, fallback_std, x),
        )

    def _class_weight_sequence(self, pred_grid: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        grid_h, grid_w = self._infer_grid_size(x)
        flat_pred = pred_grid.reshape(-1)
        patch_weight = self.classifier_weight.index_select(0, flat_pred).view(x.shape[0], grid_h * grid_w, x.shape[2])
        patch_weight = patch_weight.to(device=x.device, dtype=x.dtype)
        if self.num_extra_tokens <= 0:
            return patch_weight
        extra = torch.zeros(x.shape[0], self.num_extra_tokens, x.shape[2], device=x.device, dtype=x.dtype)
        return torch.cat((extra, patch_weight), dim=1)

    def _resolve_direction(
        self,
        diff: torch.Tensor,
        std: torch.Tensor,
        pred_grid: torch.Tensor | None,
        x: torch.Tensor,
    ) -> torch.Tensor:
        raw_direction = self._raw_direction(diff)
        if self.direction_mode == "raw_sign":
            return raw_direction
        if pred_grid is None or self.classifier_weight.abs().sum().item() == 0:
            return raw_direction
        class_weight = self._class_weight_sequence(pred_grid, x)
        weight_direction = self._safe_sign(class_weight, raw_direction)
        if self.direction_mode == "weight_sign":
            return weight_direction
        if self.direction_mode == "mixed":
            normalized_diff = diff / (std + self.eps)
            mixed_score = self.lambda_mix * normalized_diff + (1.0 - self.lambda_mix) * class_weight
            return self._safe_sign(mixed_score, raw_direction)
        raise ValueError(f"Unsupported direction mode: {self.direction_mode}")

    def _resolve_base_alpha(self, threshold: torch.Tensor | float) -> torch.Tensor | float:
        if self.alpha0_mode == "threshold":
            return threshold
        return threshold if self.alpha0 is None else self.alpha0

    def _resolve_beta(self, threshold: torch.Tensor | float) -> torch.Tensor | float:
        if self.beta_scale is None:
            return self.beta
        if torch.is_tensor(threshold):
            return threshold * float(self.beta_scale)
        return float(threshold) * float(self.beta_scale)

    def _margin_sequence(self, margin_grid: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        patch_margin = margin_grid.reshape(x.shape[0], -1)
        if self.num_extra_tokens <= 0:
            return patch_margin
        extra_margin = torch.full(
            (x.shape[0], self.num_extra_tokens),
            1.0e6,
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat((extra_margin, patch_margin), dim=1)

    def _resolve_alpha(self, threshold: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        alpha = self._expand_alpha(self._resolve_base_alpha(threshold), x)
        if self.mode != "margin_extra_sparse":
            return alpha
        margin_grid = self._resolve_margin_grid(x)
        if margin_grid is None:
            return alpha
        margin = self._margin_sequence(margin_grid.to(device=x.device, dtype=x.dtype), x)
        tau = max(self.tau, self.eps)
        adaptive = self._expand_alpha(self._resolve_beta(threshold), x) * torch.exp(-margin / tau).unsqueeze(-1)
        return alpha + adaptive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._update_statistics(x)
        if not self.runtime_enabled:
            return x
        threshold = self._resolve_threshold()
        bias, std = self._global_bias_std()
        pred_grid = self._resolve_pred_grid(x)
        if self.mode == "cc_extra_sparse":
            bias, std = self._select_class_stats(pred_grid, x, bias, std)
        crop = self._resolve_alpha(threshold, x) * std
        diff = x - bias
        if self.mode == "cc_extra_sparse":
            direction = self._raw_direction(diff)
        elif self.mode in {"dir_extra_sparse", "margin_extra_sparse"}:
            direction = self._resolve_direction(diff, std, pred_grid, x)
        else:
            raise ValueError(f"Unsupported PostSparse mode: {self.mode}")
        inside = torch.abs(diff) < crop
        self.last_soft_changed_fraction = torch.sigmoid((crop - torch.abs(diff)) / self.soft_change_temperature).mean()
        target = bias + direction * crop
        output = torch.where(inside, target, x)
        return _apply_identity_ste(x, output, self.use_bpda_ste())


class CCExtraSparseTokens(PostSparseTokens):
    def __init__(self, channels: int, *, num_classes: int = 21, num_extra_tokens: int = 1) -> None:
        super().__init__(channels, mode="cc_extra_sparse", num_classes=num_classes, num_extra_tokens=num_extra_tokens)


class DirExtraSparseTokens(PostSparseTokens):
    def __init__(self, channels: int, *, num_classes: int = 21, num_extra_tokens: int = 1) -> None:
        super().__init__(channels, mode="dir_extra_sparse", num_classes=num_classes, num_extra_tokens=num_extra_tokens)


class MarginExtraSparseTokens(PostSparseTokens):
    def __init__(self, channels: int, *, num_classes: int = 21, num_extra_tokens: int = 1) -> None:
        super().__init__(
            channels,
            mode="margin_extra_sparse",
            num_classes=num_classes,
            num_extra_tokens=num_extra_tokens,
        )


MEANSPARSE_LAYER_TYPES = (MeanSparse2d, MeanSparseTokens)
EXTRASPARSE_LAYER_TYPES = (ExtraSparse2d, ExtraSparseTokens)
POSTSPARSE_LAYER_TYPES = (PostSparse2d, PostSparseTokens)
SPARSE_LAYER_TYPES = (_RunningStatsSparse2d, _RunningStatsSparseTokens)


def iter_sparse_modules(
    model: nn.Module,
    layer_types: type[nn.Module] | tuple[type[nn.Module], ...] = SPARSE_LAYER_TYPES,
) -> Iterable[nn.Module]:
    for module in model.modules():
        if isinstance(module, layer_types):
            yield module


def iter_sparse_module_items(
    model: nn.Module,
    layer_types: type[nn.Module] | tuple[type[nn.Module], ...] = SPARSE_LAYER_TYPES,
) -> Iterable[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            yield name, module


def iter_meansparse_modules(model: nn.Module) -> Iterable[nn.Module]:
    yield from iter_sparse_modules(model, MEANSPARSE_LAYER_TYPES)


def iter_extrasparse_modules(model: nn.Module) -> Iterable[nn.Module]:
    yield from iter_sparse_modules(model, EXTRASPARSE_LAYER_TYPES)


def iter_postsparse_modules(model: nn.Module) -> Iterable[nn.Module]:
    yield from iter_sparse_modules(model, POSTSPARSE_LAYER_TYPES)


def _add_sparse_layer(
    owner: nn.Module,
    *,
    name: str,
    layer_cls: type[nn.Module],
    channels: int,
    device: torch.device,
    num_classes: int,
    num_extra_tokens: int = 1,
) -> None:
    if issubclass(layer_cls, PostSparseTokens):
        owner.add_module(name, layer_cls(channels, num_classes=num_classes, num_extra_tokens=num_extra_tokens))
    elif issubclass(layer_cls, PostSparse2d):
        owner.add_module(name, layer_cls(channels, num_classes=num_classes))
    elif issubclass(layer_cls, _RunningStatsSparseTokens):
        owner.add_module(name, layer_cls(channels, num_extra_tokens=num_extra_tokens))
    else:
        owner.add_module(name, layer_cls(channels))
    getattr(owner, name).to(device)


def _patch_convnext_stem(
    stem: ConvStem,
    *,
    layer_cls: type[_RunningStatsSparse2d],
    prefix: str,
    num_classes: int,
) -> None:
    patched_flag = f"_{prefix}_patched"
    if getattr(stem, patched_flag, False):
        return
    device = stem.stem[0].weight.device
    name1 = f"{prefix}_stem1"
    name2 = f"{prefix}_stem2"
    _add_sparse_layer(
        stem,
        name=name1,
        layer_cls=layer_cls,
        channels=stem.stem[0].out_channels,
        device=device,
        num_classes=num_classes,
    )
    _add_sparse_layer(
        stem,
        name=name2,
        layer_cls=layer_cls,
        channels=stem.stem[3].out_channels,
        device=device,
        num_classes=num_classes,
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem[0](x)
        x = self.stem[1](x)
        x = getattr(self, name1)(x)
        x = self.stem[2](x)
        x = self.stem[3](x)
        x = self.stem[4](x)
        x = getattr(self, name2)(x)
        x = self.stem[5](x)
        return x

    stem.forward = types.MethodType(forward, stem)
    setattr(stem, patched_flag, True)


def _patch_convnext_block(
    block: ConvNeXtBlock,
    *,
    layer_cls: type[_RunningStatsSparse2d],
    prefix: str,
    num_classes: int,
) -> None:
    patched_flag = f"_{prefix}_patched"
    if getattr(block, patched_flag, False):
        return
    device = block.dwconv.weight.device
    name = f"{prefix}_block"
    _add_sparse_layer(
        block,
        name=name,
        layer_cls=layer_cls,
        channels=block.dwconv.in_channels,
        device=device,
        num_classes=num_classes,
    )

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
        x = residual + self.drop_path(x)
        return getattr(self, name)(x)

    block.forward = types.MethodType(forward, block)
    setattr(block, patched_flag, True)


def _patch_convnext_backbone(
    backbone: ConvNeXt,
    *,
    layer_cls: type[_RunningStatsSparse2d],
    prefix: str,
    num_classes: int,
) -> None:
    patched_flag = f"_{prefix}_patched"
    if getattr(backbone, patched_flag, False):
        return
    if not isinstance(backbone.downsample_layers[0], ConvStem):
        raise TypeError(f"Unsupported ConvNeXt stem type: {type(backbone.downsample_layers[0])!r}")
    _patch_convnext_stem(backbone.downsample_layers[0], layer_cls=layer_cls, prefix=prefix, num_classes=num_classes)
    device = next(backbone.parameters()).device
    for stage in backbone.stages:
        for module in stage.modules():
            if isinstance(module, ConvNeXtBlock):
                _patch_convnext_block(module, layer_cls=layer_cls, prefix=prefix, num_classes=num_classes)
    for stage_idx in range(1, len(backbone.downsample_layers)):
        downsample = backbone.downsample_layers[stage_idx]
        conv = downsample[1]
        _add_sparse_layer(
            backbone,
            name=f"{prefix}_downsample{stage_idx}",
            layer_cls=layer_cls,
            channels=conv.out_channels,
            device=device,
            num_classes=num_classes,
        )

    def forward_features(
        self,
        x: torch.Tensor,
        collect_intermediates: bool = False,
    ) -> tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        outputs: list[torch.Tensor] = []
        intermediates: dict[str, torch.Tensor] = {}
        for stage_idx in range(4):
            x = self.downsample_layers[stage_idx](x)
            if stage_idx > 0:
                x = getattr(self, f"{prefix}_downsample{stage_idx}")(x)
            norm_layer = getattr(self, f"norm{stage_idx}")
            for block_idx, block in enumerate(self.stages[stage_idx]):
                x = block(x)
                if collect_intermediates:
                    intermediates[f"backbone:stage{stage_idx}:block{block_idx:02d}"] = x
            if stage_idx in self.out_indices:
                normalized = norm_layer(x)
                outputs.append(normalized)
                if collect_intermediates:
                    intermediates[f"backbone:stage{stage_idx}"] = normalized
        if collect_intermediates:
            return tuple(outputs), intermediates
        return tuple(outputs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.forward_features(x)

    backbone.forward_features = types.MethodType(forward_features, backbone)
    backbone.forward = types.MethodType(forward, backbone)
    setattr(backbone, patched_flag, True)


def _patch_resnet_stem(
    stem: nn.Sequential,
    *,
    layer_cls: type[_RunningStatsSparse2d],
    prefix: str,
    num_classes: int,
) -> None:
    patched_flag = f"_{prefix}_patched"
    if getattr(stem, patched_flag, False):
        return
    device = stem[0].weight.device
    name = f"{prefix}_stem1"
    _add_sparse_layer(
        stem,
        name=name,
        layer_cls=layer_cls,
        channels=stem[0].out_channels,
        device=device,
        num_classes=num_classes,
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self[0](x)
        x = self[1](x)
        x = getattr(self, name)(x)
        x = self[2](x)
        x = self[3](x)
        return x

    stem.forward = types.MethodType(forward, stem)
    setattr(stem, patched_flag, True)


def _patch_basic_block(
    block: BasicBlock,
    *,
    layer_cls: type[_RunningStatsSparse2d],
    prefix: str,
    num_classes: int,
) -> None:
    patched_flag = f"_{prefix}_patched"
    if getattr(block, patched_flag, False):
        return
    device = block.conv1.weight.device
    name1 = f"{prefix}1"
    name2 = f"{prefix}2"
    _add_sparse_layer(
        block,
        name=name1,
        layer_cls=layer_cls,
        channels=block.conv1.out_channels,
        device=device,
        num_classes=num_classes,
    )
    _add_sparse_layer(
        block,
        name=name2,
        layer_cls=layer_cls,
        channels=block.conv2.out_channels,
        device=device,
        num_classes=num_classes,
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = getattr(self, name1)(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = getattr(self, name2)(out)
        out = self.relu(out)
        return out

    block.forward = types.MethodType(forward, block)
    setattr(block, patched_flag, True)


def _patch_bottleneck(
    block: Bottleneck,
    *,
    layer_cls: type[_RunningStatsSparse2d],
    prefix: str,
    num_classes: int,
) -> None:
    patched_flag = f"_{prefix}_patched"
    if getattr(block, patched_flag, False):
        return
    device = block.conv1.weight.device
    name1 = f"{prefix}1"
    name2 = f"{prefix}2"
    name3 = f"{prefix}3"
    _add_sparse_layer(
        block,
        name=name1,
        layer_cls=layer_cls,
        channels=block.conv1.out_channels,
        device=device,
        num_classes=num_classes,
    )
    _add_sparse_layer(
        block,
        name=name2,
        layer_cls=layer_cls,
        channels=block.conv2.out_channels,
        device=device,
        num_classes=num_classes,
    )
    _add_sparse_layer(
        block,
        name=name3,
        layer_cls=layer_cls,
        channels=block.conv3.out_channels,
        device=device,
        num_classes=num_classes,
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = getattr(self, name1)(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = getattr(self, name2)(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = getattr(self, name3)(out)
        out = self.relu(out)
        return out

    block.forward = types.MethodType(forward, block)
    setattr(block, patched_flag, True)


def _patch_resnet_backbone(
    backbone: TorchvisionResNetBackbone,
    *,
    layer_cls: type[_RunningStatsSparse2d],
    prefix: str,
    num_classes: int,
) -> None:
    patched_flag = f"_{prefix}_patched"
    if getattr(backbone, patched_flag, False):
        return
    _patch_resnet_stem(backbone.stem, layer_cls=layer_cls, prefix=prefix, num_classes=num_classes)
    for module in backbone.modules():
        if isinstance(module, BasicBlock):
            _patch_basic_block(module, layer_cls=layer_cls, prefix=prefix, num_classes=num_classes)
        elif isinstance(module, Bottleneck):
            _patch_bottleneck(module, layer_cls=layer_cls, prefix=prefix, num_classes=num_classes)
    setattr(backbone, patched_flag, True)


def _patch_vit_block(
    block: ViTBlock,
    *,
    layer_cls: type[nn.Module],
    prefix: str,
    num_classes: int,
    num_extra_tokens: int,
) -> None:
    patched_flag = f"_{prefix}_patched"
    if getattr(block, patched_flag, False):
        return
    device = block.norm1.weight.device
    channels = int(block.norm1.normalized_shape[0])
    name1 = f"{prefix}1"
    name2 = f"{prefix}2"
    _add_sparse_layer(
        block,
        name=name1,
        layer_cls=layer_cls,
        channels=channels,
        device=device,
        num_classes=num_classes,
        num_extra_tokens=num_extra_tokens,
    )
    _add_sparse_layer(
        block,
        name=name2,
        layer_cls=layer_cls,
        channels=channels,
        device=device,
        num_classes=num_classes,
        num_extra_tokens=num_extra_tokens,
    )

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
        x = getattr(self, name1)(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = getattr(self, name2)(x)
        return x

    block.forward = types.MethodType(forward, block)
    setattr(block, patched_flag, True)


def _set_token_sparse_grid_size(encoder: VisionTransformer, grid_size: tuple[int, int]) -> None:
    for module in iter_sparse_modules(encoder, SPARSE_LAYER_TYPES):
        if isinstance(module, _RunningStatsSparseTokens):
            module.set_grid_size(grid_size)


def _patch_vit_encoder(
    encoder: VisionTransformer,
    *,
    layer_cls: type[nn.Module],
    prefix: str,
    num_classes: int,
) -> None:
    patched_flag = f"_{prefix}_patched"
    if getattr(encoder, patched_flag, False):
        return
    num_extra_tokens = 1 + int(encoder.distilled)
    for block in encoder.blocks:
        _patch_vit_block(
            block,
            layer_cls=layer_cls,
            prefix=prefix,
            num_classes=num_classes,
            num_extra_tokens=num_extra_tokens,
        )

    def forward_tokens(
        self,
        x: torch.Tensor,
        collect_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        grid_size = (x.shape[2] // self.patch_size, x.shape[3] // self.patch_size)
        _set_token_sparse_grid_size(self, grid_size)
        x = self._prepare_tokens(x)
        hidden_states: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            if collect_hidden_states:
                hidden_states.append(x)
        x = self.norm(x)
        return x, hidden_states

    encoder.forward_tokens = types.MethodType(forward_tokens, encoder)
    setattr(encoder, patched_flag, True)


def configure_postsparse_classifier_weights(model: nn.Module) -> None:
    if isinstance(model, UperNetForSemanticSegmentation):
        weight = model.decode_head.classifier.weight.detach()
    elif isinstance(model, SegMenter):
        weight = None
        if isinstance(model.decoder, DecoderLinear):
            weight = model.decoder.head.weight.detach()
        elif isinstance(model.decoder, MaskTransformer):
            cls_emb = model.decoder.cls_emb.detach().squeeze(0)
            weight = cls_emb @ model.decoder.proj_classes.detach()
    else:
        return
    for module in iter_postsparse_modules(model):
        module.set_classifier_weight(weight)


def _postsparse_margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] < 2:
        return torch.zeros(
            logits.shape[0],
            logits.shape[2],
            logits.shape[3],
            device=logits.device,
            dtype=logits.dtype,
        )
    top2 = logits.topk(k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def _set_postsparse_runtime_context(model: nn.Module, logits: torch.Tensor) -> None:
    pred = logits.argmax(dim=1)
    margin = _postsparse_margin_from_logits(logits)
    for module in iter_postsparse_modules(model):
        module.runtime_enabled = True
        module.set_runtime_context(pred=pred, logits=logits, margin=margin)


def _clear_postsparse_runtime_context(model: nn.Module) -> None:
    for module in iter_postsparse_modules(model):
        module.runtime_enabled = True
        module.clear_runtime_context()


def _configure_postsparse_modules(model: nn.Module, config: SparseDefenseConfig) -> None:
    for module in iter_postsparse_modules(model):
        module.direction_mode = config.direction_mode
        module.lambda_mix = float(config.lambda_mix)
        module.alpha0 = config.alpha0
        module.alpha0_mode = config.alpha0_mode
        module.beta = float(config.beta)
        module.beta_scale = config.beta_scale
        module.tau = float(config.tau)


def _patch_segmentor_for_postsparse_runtime(model: nn.Module) -> None:
    if getattr(model, "_postsparse_runtime_patched", False):
        return

    if isinstance(model, UperNetForSemanticSegmentation):
        original_forward = model.forward

        def forward(self, input: torch.Tensor | None = None, lbl: torch.Tensor | None = None):
            if self.training or lbl is not None:
                return original_forward(input, lbl)
            if not getattr(self, "_postsparse_runtime_enabled", True):
                return original_forward(input, lbl)
            processors = list(iter_postsparse_modules(self))
            if not processors:
                return original_forward(input, lbl)
            previous_state = [(module, module.runtime_enabled) for module in processors]
            previous_runtime_flag = getattr(self, "_postsparse_runtime_enabled", True)
            self._postsparse_runtime_enabled = False
            try:
                for module, _ in previous_state:
                    module.runtime_enabled = False
                    module.clear_runtime_context()
                with torch.no_grad():
                    reference_logits = original_forward(input, None)
            finally:
                self._postsparse_runtime_enabled = previous_runtime_flag
                for module, enabled in previous_state:
                    module.runtime_enabled = enabled
            _set_postsparse_runtime_context(self, reference_logits)
            try:
                return original_forward(input, None)
            finally:
                _clear_postsparse_runtime_context(self)

    elif isinstance(model, SegMenter):
        original_forward = model.forward

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.training:
                return original_forward(x)
            if not getattr(self, "_postsparse_runtime_enabled", True):
                return original_forward(x)
            processors = list(iter_postsparse_modules(self))
            if not processors:
                return original_forward(x)
            previous_state = [(module, module.runtime_enabled) for module in processors]
            previous_runtime_flag = getattr(self, "_postsparse_runtime_enabled", True)
            self._postsparse_runtime_enabled = False
            try:
                for module, _ in previous_state:
                    module.runtime_enabled = False
                    module.clear_runtime_context()
                with torch.no_grad():
                    reference_logits = original_forward(x)
            finally:
                self._postsparse_runtime_enabled = previous_runtime_flag
                for module, enabled in previous_state:
                    module.runtime_enabled = enabled
            _set_postsparse_runtime_context(self, reference_logits)
            try:
                return original_forward(x)
            finally:
                _clear_postsparse_runtime_context(self)

    else:
        raise TypeError(
            f"PostSparse runtime patch only supports UperNetForSemanticSegmentation or SegMenter, got {type(model)!r}"
        )

    model.forward = types.MethodType(forward, model)
    model._postsparse_runtime_enabled = True
    model._postsparse_runtime_patched = True


def _patch_model_for_variant(model: nn.Module, family: str, config: SparseDefenseConfig) -> None:
    if not supports_sparse_defense(family):
        raise NotImplementedError(
            f"Sparse defense `{config.variant}` is not implemented for family `{family}`. "
            f"Supported families: {sorted(SUPPORTED_SPARSE_FAMILIES)}."
        )
    if family in {"upernet_convnext", "upernet_resnet50"}:
        if not isinstance(model, UperNetForSemanticSegmentation):
            raise TypeError(f"Sparse defense integration expects UperNetForSemanticSegmentation, got {type(model)!r}")
        if family == "upernet_convnext":
            if not isinstance(model.backbone, ConvNeXt):
                raise TypeError(f"Expected ConvNeXt backbone for `{family}`, got {type(model.backbone)!r}")
            patch_backbone = _patch_convnext_backbone
        else:
            if not isinstance(model.backbone, TorchvisionResNetBackbone):
                raise TypeError(f"Expected TorchvisionResNetBackbone for `{family}`, got {type(model.backbone)!r}")
            patch_backbone = _patch_resnet_backbone

        if config.variant == "meansparse":
            layer_cls: type[nn.Module] = MeanSparse2d
            prefix = "meansparse"
        elif config.variant == "extrasparse":
            layer_cls = ExtraSparse2d
            prefix = "extrasparse"
        elif config.variant == "cc_extra_sparse":
            layer_cls = CCExtraSparse2d
            prefix = "cc_extrasparse"
        elif config.variant == "dir_extra_sparse":
            layer_cls = DirExtraSparse2d
            prefix = "dir_extrasparse"
        elif config.variant == "margin_extra_sparse":
            layer_cls = MarginExtraSparse2d
            prefix = "margin_extrasparse"
        else:
            raise ValueError(f"Unsupported sparse-defense variant: {config.variant}")

        patch_backbone(model.backbone, layer_cls=layer_cls, prefix=prefix, num_classes=model.decode_head.cls)
    elif family == "segmenter_vit_s":
        if not isinstance(model, SegMenter):
            raise TypeError(f"Sparse defense integration expects SegMenter, got {type(model)!r}")
        if not isinstance(model.encoder, VisionTransformer):
            raise TypeError(f"Expected VisionTransformer encoder for `{family}`, got {type(model.encoder)!r}")
        if config.variant == "meansparse":
            layer_cls = MeanSparseTokens
            prefix = "meansparse"
        elif config.variant == "extrasparse":
            layer_cls = ExtraSparseTokens
            prefix = "extrasparse"
        elif config.variant == "cc_extra_sparse":
            layer_cls = CCExtraSparseTokens
            prefix = "cc_extrasparse"
        elif config.variant == "dir_extra_sparse":
            layer_cls = DirExtraSparseTokens
            prefix = "dir_extrasparse"
        elif config.variant == "margin_extra_sparse":
            layer_cls = MarginExtraSparseTokens
            prefix = "margin_extrasparse"
        else:
            raise ValueError(f"Unsupported sparse-defense variant: {config.variant}")
        _patch_vit_encoder(model.encoder, layer_cls=layer_cls, prefix=prefix, num_classes=model.n_cls)
    else:
        raise NotImplementedError(f"Unsupported sparse-defense family: {family}")

    if config.is_postsparse:
        _patch_segmentor_for_postsparse_runtime(model)
        _configure_postsparse_modules(model, config)
        configure_postsparse_classifier_weights(model)


def _set_variant_threshold(model: nn.Module, variant: str, threshold: float) -> None:
    for module in iter_sparse_modules(model):
        if variant == "meansparse" and isinstance(module, MEANSPARSE_LAYER_TYPES):
            module.set_threshold(threshold)
        elif variant == "extrasparse" and isinstance(module, EXTRASPARSE_LAYER_TYPES):
            module.set_threshold(threshold)
        elif variant in POSTSPARSE_VARIANTS and isinstance(module, POSTSPARSE_LAYER_TYPES):
            module.set_threshold(threshold)


def _copy_tensor_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    target = getattr(module, name)
    target.copy_(value.to(device=target.device, dtype=target.dtype))


def _sparse_module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        "running_mean": module.running_mean.detach().cpu(),
        "running_var": module.running_var.detach().cpu(),
    }
    if isinstance(module, POSTSPARSE_LAYER_TYPES):
        state["class_conditional_mean"] = module.class_conditional_mean.detach().cpu()
        state["class_conditional_std"] = module.class_conditional_std.detach().cpu()
        state["class_count"] = module.class_count.detach().cpu()
    return state


def export_sparse_sidecar(
    model: nn.Module,
    *,
    family: str,
    config: SparseDefenseConfig,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    modules = {name: _sparse_module_state(module) for name, module in iter_sparse_module_items(model)}
    payload = {
        "format_version": SPARSE_SIDECAR_FORMAT_VERSION,
        "family": family,
        "variant": config.variant,
        "num_sparse_modules": len(modules),
        "metadata": dict(metadata or {}),
        "modules": modules,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


def load_sparse_sidecar(
    model: nn.Module,
    *,
    family: str,
    config: SparseDefenseConfig,
) -> dict[str, Any]:
    if config.stats_path is None:
        raise ValueError("Sparse defense config requires `stats_path` when loading stats.")
    sidecar_path = Path(config.stats_path)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Sparse stats sidecar not found: {sidecar_path}")
    try:
        payload = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(sidecar_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid sparse sidecar payload type: {type(payload)!r}")
    expected_family = payload.get("family")
    expected_variant = payload.get("variant")
    if config.strict_stats:
        if expected_family not in {None, family}:
            raise ValueError(f"Sparse stats family mismatch: expected `{family}`, found `{expected_family}`")
        if expected_variant not in {None, config.variant}:
            raise ValueError(f"Sparse stats variant mismatch: expected `{config.variant}`, found `{expected_variant}`")
    module_payload = payload.get("modules")
    if not isinstance(module_payload, dict):
        raise ValueError(f"Sparse sidecar missing `modules` mapping: {sidecar_path}")
    named_modules = dict(iter_sparse_module_items(model))
    missing_modules = sorted(set(module_payload) - set(named_modules))
    extra_modules = sorted(set(named_modules) - set(module_payload))
    if config.strict_stats and missing_modules:
        raise ValueError(f"Sparse stats reference unknown modules: {missing_modules[:5]}")
    loaded_count = 0
    for name, state in module_payload.items():
        module = named_modules.get(name)
        if module is None:
            continue
        if not isinstance(state, dict):
            raise ValueError(f"Invalid sparse state for module `{name}`: {type(state)!r}")
        for key in ("running_mean", "running_var"):
            tensor = state.get(key)
            if tensor is None:
                if config.strict_stats:
                    raise ValueError(f"Sparse stats missing `{key}` for module `{name}`")
                continue
            _copy_tensor_buffer(module, key, tensor)
        if isinstance(module, POSTSPARSE_LAYER_TYPES):
            class_mean = state.get("class_conditional_mean")
            class_std = state.get("class_conditional_std")
            class_count = state.get("class_count")
            if class_mean is None or class_std is None:
                if config.strict_stats:
                    raise ValueError(f"PostSparse stats missing class statistics for module `{name}`")
            else:
                module.set_class_statistics(class_mean, class_std, class_count)
        loaded_count += 1
    return {
        "path": str(sidecar_path.resolve()),
        "loaded_modules": loaded_count,
        "missing_modules": missing_modules,
        "extra_modules": extra_modules,
        "metadata": payload.get("metadata", {}),
    }


def apply_sparse_defense(
    model: nn.Module,
    *,
    family: str,
    config: SparseDefenseConfig,
    load_stats: bool = True,
) -> dict[str, Any]:
    if config.family is not None and config.family != family:
        raise ValueError(f"Sparse defense config targets family `{config.family}`, but loader received `{family}`")
    _patch_model_for_variant(model, family, config)
    sidecar_info: dict[str, Any] | None = None
    if load_stats:
        sidecar_info = load_sparse_sidecar(model, family=family, config=config)
    _set_variant_threshold(model, config.variant, config.threshold)
    if config.is_postsparse:
        _configure_postsparse_modules(model, config)
        configure_postsparse_classifier_weights(model)
    return {
        "variant": config.variant,
        "threshold": float(config.threshold),
        "stats_path": None if config.stats_path is None else str(config.stats_path),
        "sidecar": sidecar_info,
    }


def _start_sparse_statistics(
    model: nn.Module,
    layer_types: type[nn.Module] | tuple[type[nn.Module], ...],
    num_batches: int,
) -> None:
    for module in iter_sparse_modules(model, layer_types):
        module.start_statistics(num_batches)


def _stop_sparse_statistics(
    model: nn.Module,
    layer_types: type[nn.Module] | tuple[type[nn.Module], ...],
) -> None:
    for module in iter_sparse_modules(model, layer_types):
        module.stop_statistics()


def _reset_sparse_statistics(
    model: nn.Module,
    layer_types: type[nn.Module] | tuple[type[nn.Module], ...],
) -> None:
    for module in iter_sparse_modules(model, layer_types):
        module.reset_statistics()


def _snapshot_sparse_thresholds(
    model: nn.Module,
    layer_types: type[nn.Module] | tuple[type[nn.Module], ...],
) -> list[tuple[nn.Module, float]]:
    snapshot: list[tuple[nn.Module, float]] = []
    for module in iter_sparse_modules(model, layer_types):
        snapshot.append((module, float(module.threshold.item())))
    return snapshot


def _restore_sparse_thresholds(snapshot: list[tuple[nn.Module, float]]) -> None:
    for module, threshold in snapshot:
        module.set_threshold(threshold)


class _PostSparseStatsCollector:
    def __init__(self, layer_name: str, num_features: int, num_classes: int, ignore_index: int | None) -> None:
        self.layer_name = layer_name
        self.num_features = num_features
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_sum = torch.zeros(num_classes, num_features, dtype=torch.float64)
        self.class_sq_sum = torch.zeros(num_classes, num_features, dtype=torch.float64)
        self.class_count = torch.zeros(num_classes, dtype=torch.long)

    def update(
        self,
        activation: torch.Tensor,
        targets: torch.Tensor,
        *,
        num_extra_tokens: int = 0,
        grid_size: tuple[int, int] | None = None,
    ) -> None:
        if activation.ndim == 4:
            if targets.shape[-2:] != activation.shape[-2:]:
                targets = F.interpolate(
                    targets.unsqueeze(1).float(),
                    size=activation.shape[-2:],
                    mode="nearest",
                ).squeeze(1).long()
            flat_activation = activation.detach().permute(0, 2, 3, 1).reshape(-1, activation.shape[1]).to(dtype=torch.float64)
        elif activation.ndim == 3:
            if grid_size is None:
                patch_tokens = activation.shape[1] - num_extra_tokens
                side = int(round(patch_tokens**0.5))
                if side * side != patch_tokens:
                    raise ValueError(f"Cannot infer grid size for token activation shape {tuple(activation.shape)}")
                grid_size = (side, side)
            grid_h, grid_w = grid_size
            patch_tokens = activation[:, num_extra_tokens:]
            if patch_tokens.shape[1] != grid_h * grid_w:
                raise ValueError(
                    f"Token activation shape mismatch for {self.layer_name}: "
                    f"{patch_tokens.shape[1]} patch tokens vs grid {grid_h}x{grid_w}"
                )
            if targets.shape[-2:] != grid_size:
                targets = F.interpolate(targets.unsqueeze(1).float(), size=grid_size, mode="nearest").squeeze(1).long()
            flat_activation = patch_tokens.detach().reshape(-1, patch_tokens.shape[-1]).to(dtype=torch.float64)
        else:
            raise ValueError(f"Unsupported activation rank for {self.layer_name}: {activation.ndim}")

        if self.ignore_index is None:
            valid = torch.ones_like(targets, dtype=torch.bool)
        else:
            valid = targets != self.ignore_index
        if not torch.any(valid):
            return
        flat_targets = targets.reshape(-1)
        flat_valid = valid.reshape(-1)
        valid_targets = flat_targets[flat_valid]
        valid_activation = flat_activation[flat_valid]
        for class_index in torch.unique(valid_targets).tolist():
            class_mask = valid_targets == int(class_index)
            class_activation = valid_activation[class_mask]
            self.class_sum[class_index] += class_activation.sum(dim=0).cpu()
            self.class_sq_sum[class_index] += class_activation.square().sum(dim=0).cpu()
            self.class_count[class_index] += int(class_mask.sum().item())

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        safe_count = torch.clamp(self.class_count.to(dtype=torch.float64).unsqueeze(1), min=1.0)
        mean = self.class_sum / safe_count
        var = torch.clamp(self.class_sq_sum / safe_count - mean.square(), min=0.0)
        return mean.float(), torch.sqrt(var).float(), self.class_count.clone()


def calibrate_sparse_defense(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    config: SparseDefenseConfig,
    ignore_index: int | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    if config.variant == "meansparse":
        modules = list(iter_meansparse_modules(model))
        if not modules:
            raise ValueError("Model does not contain MeanSparse modules.")
        threshold_snapshot = _snapshot_sparse_thresholds(model, MEANSPARSE_LAYER_TYPES)
        _reset_sparse_statistics(model, MEANSPARSE_LAYER_TYPES)
        _start_sparse_statistics(model, MEANSPARSE_LAYER_TYPES, len(dataloader))
        try:
            _set_variant_threshold(model, config.variant, 0.0)
            with torch.no_grad():
                for images, *_ in dataloader:
                    model(images.to(device))
        finally:
            _stop_sparse_statistics(model, MEANSPARSE_LAYER_TYPES)
            _restore_sparse_thresholds(threshold_snapshot)
        return {"variant": config.variant, "num_batches": len(dataloader), "num_sparse_modules": len(modules)}

    if config.variant == "extrasparse":
        modules = list(iter_extrasparse_modules(model))
        if not modules:
            raise ValueError("Model does not contain ExtraSparse modules.")
        threshold_snapshot = _snapshot_sparse_thresholds(model, EXTRASPARSE_LAYER_TYPES)
        _reset_sparse_statistics(model, EXTRASPARSE_LAYER_TYPES)
        _start_sparse_statistics(model, EXTRASPARSE_LAYER_TYPES, len(dataloader))
        try:
            _set_variant_threshold(model, config.variant, 0.0)
            with torch.no_grad():
                for images, *_ in dataloader:
                    model(images.to(device))
        finally:
            _stop_sparse_statistics(model, EXTRASPARSE_LAYER_TYPES)
            _restore_sparse_thresholds(threshold_snapshot)
        return {"variant": config.variant, "num_batches": len(dataloader), "num_sparse_modules": len(modules)}

    modules = {name: module for name, module in iter_sparse_module_items(model, POSTSPARSE_LAYER_TYPES)}
    if not modules:
        raise ValueError("Model does not contain PostSparse modules.")
    threshold_snapshot = _snapshot_sparse_thresholds(model, POSTSPARSE_LAYER_TYPES)
    _reset_sparse_statistics(model, POSTSPARSE_LAYER_TYPES)
    _start_sparse_statistics(model, POSTSPARSE_LAYER_TYPES, len(dataloader))
    collectors = {
        name: _PostSparseStatsCollector(
            layer_name=name,
            num_features=module.running_mean.numel(),
            num_classes=module.num_classes,
            ignore_index=ignore_index,
        )
        for name, module in modules.items()
    }
    current_targets: dict[str, torch.Tensor | None] = {"value": None}
    handles = []

    for name, module in modules.items():
        collector = collectors[name]

        def hook(_module, _inputs, output, collector: _PostSparseStatsCollector = collector):
            targets = current_targets["value"]
            if targets is None:
                raise RuntimeError(f"Missing targets for postsparse collector `{collector.layer_name}`")
            if isinstance(_module, PostSparseTokens):
                collector.update(
                    output,
                    targets,
                    num_extra_tokens=_module.num_extra_tokens,
                    grid_size=_module._infer_grid_size(output),
                )
            else:
                collector.update(output, targets)
            return output

        handles.append(module.register_forward_hook(hook))

    previous_runtime_flag = getattr(model, "_postsparse_runtime_enabled", True)
    if hasattr(model, "_postsparse_runtime_enabled"):
        model._postsparse_runtime_enabled = False
    for module in modules.values():
        module.runtime_enabled = False
        module.clear_runtime_context()

    try:
        _set_variant_threshold(model, config.variant, 0.0)
        with torch.no_grad():
            for images, targets, *_ in dataloader:
                current_targets["value"] = targets.to(device)
                model(images.to(device))
    finally:
        for handle in handles:
            handle.remove()
        if hasattr(model, "_postsparse_runtime_enabled"):
            model._postsparse_runtime_enabled = previous_runtime_flag
        _stop_sparse_statistics(model, POSTSPARSE_LAYER_TYPES)
        _restore_sparse_thresholds(threshold_snapshot)

    for name, module in modules.items():
        class_mean, class_std, class_count = collectors[name].finalize()
        module.set_class_statistics(class_mean, class_std, class_count)
        module.runtime_enabled = True
        module.clear_runtime_context()
    configure_postsparse_classifier_weights(model)
    return {"variant": config.variant, "num_batches": len(dataloader), "num_sparse_modules": len(modules)}
