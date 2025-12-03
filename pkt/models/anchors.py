"""Anchor generation utilities for BEV feature maps."""
from __future__ import annotations

from typing import Sequence

import torch


class Anchor3DRangeGenerator:
    def __init__(self, custom_values: Sequence[float] | None = None) -> None:
        self.custom_values = list(custom_values or [])


class AlignedAnchor3DRangeGenerator(Anchor3DRangeGenerator):
    """Generate anchors whose centers align with voxel grid centers."""

    def __init__(self, align_corner: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.align_corner = align_corner

    def anchors_single_range(
        self,
        feature_size,
        anchor_range,
        scale: float = 1.0,
        sizes: Sequence[Sequence[float]] = ((3.9, 1.6, 1.56),),
        rotations: Sequence[float] = (0.0, 1.5707963),
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        if len(feature_size) == 2:
            feature_size = (1, feature_size[0], feature_size[1])
        if len(feature_size) != 3:
            raise ValueError("feature_size must be a sequence of length 2 or 3")

        z_size, y_size, x_size = int(feature_size[0]), int(feature_size[1]), int(feature_size[2])
        x_min, y_min, z_min, x_max, y_max, z_max = anchor_range

        def _centers(start, end, steps):
            pts = torch.linspace(start, end, steps + 1, device=device)
            pts = pts[:-1]
            if not self.align_corner and steps > 0:
                pts = pts + (pts[1] - pts[0]) / 2 if steps > 1 else pts + (end - start) / 2
            return pts

        z_centers = _centers(z_min, z_max, z_size)
        y_centers = _centers(y_min, y_max, y_size)
        x_centers = _centers(x_min, x_max, x_size)

        rotations_tensor = torch.tensor(rotations, device=device)
        size_tensor = torch.tensor(sizes, device=device).view(-1, 3) * scale

        xg, yg, zg, rg = torch.meshgrid(x_centers, y_centers, z_centers, rotations_tensor, indexing="ij")
        num_sizes = size_tensor.shape[0]
        num_rots = rotations_tensor.numel()

        xg = xg[..., None].expand(x_size, y_size, z_size, num_rots, num_sizes)
        yg = yg[..., None].expand_as(xg)
        zg = zg[..., None].expand_as(xg)
        rg = rg[..., None].expand_as(xg)

        sizes_expanded = size_tensor.view(1, 1, 1, 1, num_sizes, 3).expand(
            x_size, y_size, z_size, num_rots, num_sizes, 3
        )
        sizes_expanded = sizes_expanded.permute(0, 1, 2, 4, 3, 5)

        anchors = torch.stack(
            [
                xg.permute(0, 1, 2, 4, 3),
                yg.permute(0, 1, 2, 4, 3),
                zg.permute(0, 1, 2, 4, 3),
                sizes_expanded[..., 0],
                sizes_expanded[..., 1],
                sizes_expanded[..., 2],
                rg.permute(0, 1, 2, 4, 3),
            ],
            dim=-1,
        )
        anchors = anchors.permute(2, 1, 0, 3, 4, 5)

        if self.custom_values:
            custom = anchors.new_zeros(*anchors.shape[:-1], len(self.custom_values))
            anchors = torch.cat([anchors, custom], dim=-1)
        return anchors


__all__ = ["Anchor3DRangeGenerator", "AlignedAnchor3DRangeGenerator"]
