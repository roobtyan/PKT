from __future__ import annotations

from typing import Any

import torch
from torch import nn

from pkt.heads import HEAD_REGISTRY, BaseHead


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

    def forward(self, features: torch.Tensor, target: torch.Tensor | None = None) -> dict[str, Any]:
        logits = self.fc(features)
        preds = torch.argmax(logits, dim=1)
        return {"logits": logits, "preds": preds}
