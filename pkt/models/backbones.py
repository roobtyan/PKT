"""Model backbone implementations."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from pkt.registry import Registry


BACKBONE_REGISTRY = Registry("backbone")


@BACKBONE_REGISTRY.register("mlp")
class MLPBackbone(nn.Module):
    """A simple multi-layer perceptron backbone."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        layers: list[nn.Module] = []
        in_dim = input_dim
        activation_layer = _resolve_activation(activation)
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(activation_layer())
            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.output_dim = in_dim
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(x)


def _resolve_activation(name: str) -> type[nn.Module]:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in activations:
        raise KeyError(f"Unsupported activation '{name}'. Available: {', '.join(sorted(activations))}")
    return activations[name]


__all__ = ["BACKBONE_REGISTRY", "MLPBackbone"]
