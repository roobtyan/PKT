"""Feature grid visualization utilities."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pkt.engine.registries import VISUALIZERS


def normalize_feature(channel: torch.Tensor) -> torch.Tensor:
    c_min, c_max = channel.min(), channel.max()
    denom = (c_max - c_min).clamp(min=1e-6)
    return (channel - c_min) / denom


def plot_feature_grid(
    images: Sequence[Tuple[str, torch.Tensor]],
    feature_maps: Sequence[Tuple[int, torch.Tensor]],
    *,
    channels: int,
    cols: int,
    output: Path,
) -> None:
    num_views = len(images)
    rows_images = math.ceil(num_views / cols)

    rows_features = 0
    meta = []
    for stride, fmap in feature_maps:
        if fmap.dim() == 4:
            n, c, h, w = fmap.shape
        else:
            n, c, h, w = 1, fmap.shape[0], fmap.shape[1], fmap.shape[2]
        plot_c = min(channels, c)
        total_subplots = n * plot_c
        level_rows = math.ceil(total_subplots / cols)
        rows_features += level_rows
        meta.append((stride, n, plot_c))

    rows_total = rows_images + rows_features
    fig = plt.figure(figsize=(cols * 3, rows_total * 2.5))
    grid = fig.add_gridspec(rows_total, cols)

    for idx, (name, img) in enumerate(images):
        r = idx // cols
        c = idx % cols
        ax_img = fig.add_subplot(grid[r, c])
        img_np = img.permute(1, 2, 0).detach().cpu().clamp(0.0, 1.0).numpy()
        ax_img.imshow(img_np)
        ax_img.set_title(f"{name}", fontsize=8)
        ax_img.axis("off")

    row_cursor = rows_images
    for (stride, fmap), (_, n_views, n_channels) in zip(feature_maps, meta):
        fmap = fmap.detach().cpu()
        if fmap.dim() == 3:
            fmap = fmap.unsqueeze(0)
        cnt = 0
        for v in range(n_views):
            view_name = images[v][0] if v < len(images) else f"View{v}"
            for ch in range(n_channels):
                r = row_cursor + cnt // cols
                c = cnt % cols
                ax = fig.add_subplot(grid[r, c])
                feat_data = fmap[v, ch]
                normalized = normalize_feature(feat_data).numpy()
                ax.imshow(normalized, cmap="viridis")
                ax.set_title(f"S{stride}-{view_name}-C{ch}", fontsize=6)
                ax.axis("off")
                cnt += 1
        row_cursor += math.ceil(cnt / cols)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


@VISUALIZERS.register("FeatureGridVisualizer")
class FeatureGridVisualizer:
    def __init__(self, output: str, channels: int = 8, cols: int = 4) -> None:
        self.output = Path(output).expanduser()
        self.channels = channels
        self.cols = cols

    def __call__(self, data: dict[str, object]) -> Path:
        images = data.get("images")
        feature_maps = data.get("feature_maps")
        if not isinstance(images, Sequence) or not isinstance(feature_maps, Sequence):
            raise TypeError("FeatureGridVisualizer expects 'images' and 'feature_maps' sequences in data dict")
        plot_feature_grid(
            images,
            feature_maps,
            channels=self.channels,
            cols=self.cols,
            output=self.output,
        )
        return self.output


__all__ = ["FeatureGridVisualizer", "plot_feature_grid", "normalize_feature"]
