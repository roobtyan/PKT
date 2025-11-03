"""Loss function registry and implementations."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from pkt.registry import Registry
from pkt.losses.routing import (
    RoutingTrajectoryObjective,
    STLonObjective,
    TrajectoryClassificationObjective,
)

LOSS_REGISTRY = Registry("loss")

LOSS_REGISTRY.register("routing_trajectory")(RoutingTrajectoryObjective)
LOSS_REGISTRY.register("trajectory_classification")(TrajectoryClassificationObjective)
LOSS_REGISTRY.register("st_lon")(STLonObjective)


@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss with optional label smoothing and class weights."""

    def __init__(
            self,
            weight: float = 1.0,
            label_smoothing: float = 0.0,
            reduction: str = "mean",
            ignore_index: int = -100,
            class_weights: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"Invalid reduction '{reduction}'. Supported reductions are 'mean', 'sum', and 'none'."
            )
        if label_smoothing < 0:
            raise ValueError("label_smoothing must be non-negative")
        self.loss_weight = float(weight)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        self.ignore_index = int(ignore_index)
        if isinstance(class_weights, torch.Tensor):
            weight_tensor = class_weights.to(dtype=torch.float32)
        elif class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
        else:
            weight_tensor = torch.tensor([], dtype=torch.float32)
        self.register_buffer("_class_weights", weight_tensor, persistent=True)

    def forward(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
            **_: object,
    ) -> torch.Tensor:  # type: ignore[override]
        weight = self._class_weights if self._class_weights.numel() > 0 else None
        loss = F.cross_entropy(
            logits,
            target,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
        if self.loss_weight != 1.0:
            loss = loss * self.loss_weight
        return loss


__all__ = [
    "LOSS_REGISTRY",
    "CrossEntropyLoss",
    "RoutingTrajectoryObjective",
    "TrajectoryClassificationObjective",
    "STLonObjective",
]
