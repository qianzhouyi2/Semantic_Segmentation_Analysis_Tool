from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.architectures.segmenter import SegMenter, pad_to_patch_size, remove_padding
from src.models.architectures.upernet import UperNetForSemanticSegmentation


@dataclass(slots=True)
class ModelBatch:
    images: torch.Tensor
    masks: torch.Tensor | None = None


class SegmentationModelAdapter(ABC):
    """Minimal adapter interface used by adversarial attacks."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def logits(self, images: torch.Tensor) -> torch.Tensor:
        """Return dense segmentation logits with shape [B, C, H, W]."""
        raise NotImplementedError

    def forward_with_features(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.logits(images), {}

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.logits(images).argmax(dim=1)


class TorchSegmentationModelAdapter(SegmentationModelAdapter):
    """Wrap a plain torch segmentation model for attack/evaluation code."""

    def __init__(self, model: torch.nn.Module, num_classes: int, device: str | torch.device = "cpu") -> None:
        self.model = model
        self._num_classes = int(num_classes)
        self._device = torch.device(device)
        self.model.to(self._device)
        self.model.eval()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def device(self) -> torch.device:
        return self._device

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images.to(self._device))

    def forward_with_features(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model_input = images.to(self._device)

        if isinstance(self.model, UperNetForSemanticSegmentation):
            features = self.model.backbone(model_input)
            logits = self.model.decode_head(features)
            logits = F.interpolate(logits, size=model_input.shape[2:], mode="bilinear", align_corners=False)
            return logits, {
                "backbone:last": features[-1],
                "backbone:first": features[0],
            }

        if isinstance(self.model, SegMenter):
            original_size = (model_input.size(2), model_input.size(3))
            padded = pad_to_patch_size(model_input, self.model.patch_size)
            padded_size = (padded.size(2), padded.size(3))
            tokens = self.model.encoder(padded, pre_neck=True)
            num_extra_tokens = 0 if "SAM" in self.model.backbone else 1 + int(self.model.encoder.distilled)
            spatial_tokens = tokens[:, num_extra_tokens:]
            grid_h = padded_size[0] // self.model.patch_size
            grid_w = padded_size[1] // self.model.patch_size
            feature_map = spatial_tokens.transpose(1, 2).reshape(
                spatial_tokens.size(0),
                spatial_tokens.size(2),
                grid_h,
                grid_w,
            )
            logits = self.model.decoder(spatial_tokens, padded_size)
            logits = F.interpolate(logits, size=padded_size, mode="bilinear")
            logits = remove_padding(logits, original_size)
            return logits, {"encoder": feature_map}

        logits = self.logits(model_input)
        return logits, {"logits": logits}
