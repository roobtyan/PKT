"""BEV heatmap visualization utilities."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pkt.engine.registries import VISUALIZERS


def _normalize_channel(channel: torch.Tensor) -> torch.Tensor:
    c_min, c_max = channel.min(), channel.max()
    denom = (c_max - c_min).clamp(min=1e-6)
    return (channel - c_min) / denom


@VISUALIZERS.register("BEVHeatmapVisualizer")
class BEVHeatmapVisualizer:
    def __init__(
        self,
        output: str,
        reduction: str = "mean",
        channel: int = 0,
        colormap: str = "magma",
        figsize: tuple[float, float] = (6, 5),
    ) -> None:
        if reduction not in {"mean", "max"}:
            raise ValueError("reduction must be 'mean' or 'max'")
        self.output = Path(output).expanduser()
        self.reduction = reduction
        self.channel = channel
        self.colormap = colormap
        self.figsize = figsize

    def __call__(self, data: dict[str, object]) -> Path:
        bev_feats = data.get("bev_feats")
        if not isinstance(bev_feats, torch.Tensor):
            raise TypeError("BEVHeatmapVisualizer expects 'bev_feats' tensor in data dict")
        if bev_feats.dim() != 5:
            raise ValueError(
                f"BEV tensor must have shape (B, C, D, H, W); got {tuple(bev_feats.shape)}"
            )
        if self.reduction == "mean":
            bev_map = bev_feats.mean(dim=2)
        else:
            bev_map = bev_feats.max(dim=2).values
        channel = max(0, min(self.channel, bev_map.shape[1] - 1))
        bev_slice = bev_map[0, channel].detach().cpu()
        bev_norm = _normalize_channel(bev_slice).numpy()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=self.figsize)
        plt.imshow(bev_norm, cmap=self.colormap)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(self.output, dpi=200, bbox_inches="tight")
        plt.close()
        return self.output


__all__ = ["BEVHeatmapVisualizer"]
