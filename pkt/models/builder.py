"""Model building utilities."""
from __future__ import annotations

from torch import nn

from pkt.config import ModelConfig
from pkt.models.backbones import BACKBONE_REGISTRY
from pkt.heads import HEAD_REGISTRY, BaseHead


class ConfigurableModel(nn.Module):
    """Simple wrapper around backbone and head."""

    def __init__(self, backbone: nn.Module, head: BaseHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, inputs, target=None):  # type: ignore[override]
        features = self.backbone(inputs)
        return self.head(features, target)


def build_model(cfg: ModelConfig) -> ConfigurableModel:
    backbone = BACKBONE_REGISTRY.build(vars(cfg.backbone))
    params = dict(cfg.head.params)
    # ensure head knows backbone output dimension if not provided
    params.setdefault("in_features", getattr(backbone, "output_dim", None))
    if params["in_features"] is None:
        raise ValueError(
            "Head configuration must provide 'in_features' or the backbone must expose 'output_dim'"
        )
    head = HEAD_REGISTRY.build({"name": cfg.head.name, "params": params})
    return ConfigurableModel(backbone=backbone, head=head)


__all__ = ["ConfigurableModel", "build_model"]
