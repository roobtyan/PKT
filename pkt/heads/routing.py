from __future__ import annotations

from typing import Any, Dict

import torch

from pkt.heads import HEAD_REGISTRY, BaseHead


@HEAD_REGISTRY.register("routing_head")
class RoutingTrajectoryHead(BaseHead):
    """Pass-through head that forwards backbone outputs."""

    def __init__(self, in_features: int | None = None, **_: Any) -> None:
        super().__init__()
        self.in_features = in_features

    def forward(self, features: Dict[str, Any], target: torch.Tensor | None = None) -> Dict[str, Any]:
        if not isinstance(features, dict):
            raise TypeError("routing_head expects backbone to return a mapping of outputs.")
        return features
