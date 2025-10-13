"""Model head implementations."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F

from pkt.registry import Registry


HEAD_REGISTRY = Registry("head")


class BaseHead(nn.Module):
    """Base head returning dictionaries for flexibility."""

    def forward(self, features: torch.Tensor, target: torch.Tensor | None = None) -> Dict[str, Any]:
        raise NotImplementedError


@HEAD_REGISTRY.register("classification_head")
class ClassificationHead(BaseHead):
    """Linear classification layer with optional label smoothing."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.label_smoothing = label_smoothing

    def forward(self, features: torch.Tensor, target: torch.Tensor | None = None) -> Dict[str, Any]:
        logits = self.fc(features)
        preds = torch.argmax(logits, dim=1)
        output: Dict[str, Any] = {"logits": logits, "preds": preds}
        if target is not None:
            loss = F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
            output["loss"] = loss
        return output


__all__ = ["HEAD_REGISTRY", "BaseHead", "ClassificationHead"]
