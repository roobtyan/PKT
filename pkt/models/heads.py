"""Model head implementations."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from pkt.registry import Registry


HEAD_REGISTRY = Registry("head")


class BaseHead(nn.Module):
    """Base head returning dictionaries for flexibility."""

    def forward(self, features: torch.Tensor, target: torch.Tensor | None = None) -> Dict[str, Any]:
        raise NotImplementedError


@HEAD_REGISTRY.register("classification_head")
class ClassificationHead(BaseHead):
    """Linear classification layer that produces logits and predictions."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=bias)

    def forward(self, features: torch.Tensor, target: torch.Tensor | None = None) -> Dict[str, Any]:
        logits = self.fc(features)
        preds = torch.argmax(logits, dim=1)
        return {"logits": logits, "preds": preds}


__all__ = ["HEAD_REGISTRY", "BaseHead", "ClassificationHead"]
