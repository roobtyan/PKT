"""3D voxel center visualization utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pkt.engine.registries import VISUALIZERS


def _flatten_world_points(world_points: torch.Tensor) -> torch.Tensor:
    """Return (N,3) tensor of xyz coordinates."""
    if world_points.dim() != 5 or world_points.shape[-1] != 4:
        raise ValueError("world_points must have shape (B, D, H, W, 4)")
    coords = world_points[..., :3]
    return coords.reshape(-1, 3)


def _subsample_points(points: torch.Tensor, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points.cpu().numpy()
    idx = torch.linspace(0, points.shape[0] - 1, steps=max_points, device=points.device).long()
    return points.index_select(0, idx).cpu().numpy()


def plot_voxel_points(
    points: np.ndarray,
    output: Path,
    *,
    elev: float,
    azim: float,
    figsize: Sequence[float],
    point_size: float,
    color_by_height: bool,
) -> Path:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if color_by_height and points.shape[0] > 0:
        colors = points[:, 2]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, cmap="viridis", s=point_size, alpha=0.6)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="royalblue", s=point_size, alpha=0.6)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("BEV Voxel Centers")
    ax.set_box_aspect([1, 1, 0.4])
    ax.grid(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


@VISUALIZERS.register("VoxelGridVisualizer")
class VoxelGridVisualizer:
    """Render sampled world-grid points as a 3D scatter plot."""

    def __init__(
        self,
        output: str,
        max_points: int = 20000,
        elev: float = 20.0,
        azim: float = -60.0,
        figsize: Sequence[float] = (6.0, 6.0),
        point_size: float = 3.0,
        color_by_height: bool = True,
    ) -> None:
        self.output = Path(output).expanduser()
        self.max_points = max(1, int(max_points))
        self.elev = float(elev)
        self.azim = float(azim)
        self.figsize = tuple(figsize)
        self.point_size = float(point_size)
        self.color_by_height = bool(color_by_height)

    def __call__(self, data: dict[str, object]) -> Path:
        world_points = data.get("world_points")
        if not isinstance(world_points, torch.Tensor):
            raise TypeError("VoxelGridVisualizer expects 'world_points' tensor in data dict")
        flattened = _flatten_world_points(world_points.detach().cpu())
        sampled = _subsample_points(flattened, self.max_points)
        return plot_voxel_points(
            sampled,
            self.output,
            elev=self.elev,
            azim=self.azim,
            figsize=self.figsize,
            point_size=self.point_size,
            color_by_height=self.color_by_height,
        )


__all__ = ["VoxelGridVisualizer", "plot_voxel_points"]

