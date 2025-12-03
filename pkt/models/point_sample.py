"""Point sampling utilities for projecting multi-view features onto a BEV grid."""
from __future__ import annotations

from collections.abc import Mapping
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pkt.data.nuscenes import ProjectionBatch, ProjectionSample
from pkt.engine.registries import MODULES
from pkt.models.anchors import Anchor3DRangeGenerator, AlignedAnchor3DRangeGenerator


def _list_int_size(shape: Sequence[int]) -> List[int]:
    return [int(s) for s in shape]


class PointSample(nn.Module):
    """Base class that projects voxel grids through LiDAR-to-image matrices."""

    def __init__(
        self,
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        num_cams: int,
        embed_dims: int,
        order: str = "NCDHW",
        is_trace: bool = False,
        anchor_generator: Anchor3DRangeGenerator | Mapping | None = None,
        n_voxels: Sequence[int] | None = None,
        div: int = 1,
    ) -> None:
        super().__init__()
        if len(point_cloud_range) != 6:
            raise ValueError("point_cloud_range must have 6 values [x_min, y_min, z_min, x_max, y_max, z_max]")
        if len(voxel_size) != 3:
            raise ValueError("voxel_size must contain 3 entries")

        pc = torch.tensor(point_cloud_range, dtype=torch.float32)
        vs = torch.tensor(voxel_size, dtype=torch.float32)
        span = pc[3:] - pc[:3]
        if torch.any(span <= 0):
            raise ValueError("point_cloud_range must have max > min on all axes")

        div = max(1, int(div))
        if n_voxels is not None:
            if len(n_voxels) != 3:
                raise ValueError("n_voxels must have 3 values [nx, ny, nz]")
            nx, ny, nz = [max(1, int(v)) for v in n_voxels]
        else:
            nx = max(int(torch.round(span[0] / vs[0]).item()), 1)
            ny = max(int(torch.round(span[1] / vs[1]).item()), 1)
            nz = max(int(torch.round(span[2] / vs[2]).item()), 1)

        nx = max(1, nx // div)
        ny = max(1, ny // div)

        self.anchor_generator = self._init_anchor_generator(anchor_generator)
        self.point_cloud_range = pc
        self.voxel_size = vs
        self.n_voxels = [nx, ny, nz]
        self.voxel_shape = (self.n_voxels[2], self.n_voxels[1], self.n_voxels[0])  # (D, H, W)
        self.embed_dims = int(embed_dims)
        self.num_cams = int(num_cams)
        self.order = order
        self.is_trace = is_trace

        world_grid = self._create_anchor_world_grid() if self.anchor_generator else self._create_world_grid()
        self.register_buffer("world_grid", world_grid, persistent=False)

    @staticmethod
    def _init_anchor_generator(
        cfg: Anchor3DRangeGenerator | Mapping | None,
    ) -> Anchor3DRangeGenerator | None:
        if cfg is None:
            return None
        if isinstance(cfg, Anchor3DRangeGenerator):
            return cfg
        if isinstance(cfg, Mapping):
            cfg_dict = dict(cfg)
            obj_type = cfg_dict.pop("type", cfg_dict.pop("name", None))
            if obj_type is None:
                raise KeyError("anchor_generator config must include 'type'")
            if obj_type == "AlignedAnchor3DRangeGenerator":
                cls = AlignedAnchor3DRangeGenerator
            elif obj_type == "Anchor3DRangeGenerator":
                cls = Anchor3DRangeGenerator
            else:
                raise KeyError(f"Unsupported anchor generator type '{obj_type}'")
            return cls(**cfg_dict)
        raise TypeError("anchor_generator must be None, a Mapping, or an Anchor3DRangeGenerator instance")

    def _create_world_grid(self) -> torch.Tensor:
        pc = self.point_cloud_range
        nx, ny, nz = self.n_voxels
        x = torch.linspace(pc[0], pc[3], steps=nx)
        y = torch.linspace(pc[1], pc[4], steps=ny)
        z = torch.linspace(pc[2], pc[5], steps=nz)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack([xx, yy, zz], dim=-1)
        ones = torch.ones_like(xx)[..., None]
        return torch.cat([coords, ones], dim=-1)  # (D, H, W, 4)

    def _create_anchor_world_grid(self) -> torch.Tensor:
        if self.anchor_generator is None:
            raise RuntimeError("anchor generator not initialized")
        feature_size = self.voxel_shape
        anchor_range = self.point_cloud_range.tolist()
        device = self.point_cloud_range.device
        anchors = self.anchor_generator.anchors_single_range(
            feature_size=feature_size,
            anchor_range=anchor_range,
            device=device,
        )
        centers = anchors[..., 0, 0, :3]
        ones = torch.ones_like(centers[..., :1])
        return torch.cat([centers, ones], dim=-1)

    def _prepare_projection(
        self,
        proj_mats: ProjectionBatch | ProjectionSample | torch.Tensor,
        intrinsics: torch.Tensor | None,
        imgs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(proj_mats, ProjectionBatch):
            proj = proj_mats.proj_mats
            sizes = proj_mats.image_sizes
            intr = proj_mats.intrinsics
        elif isinstance(proj_mats, ProjectionSample):
            proj = proj_mats.proj_mats.unsqueeze(0)
            sizes = proj_mats.image_sizes.unsqueeze(0)
            intr = proj_mats.intrinsics.unsqueeze(0)
        else:
            if intrinsics is None or imgs is None:
                raise ValueError("Raw projection matrices require accompanying intrinsics and imgs")
            proj = proj_mats
            if proj.dim() == 3:
                proj = proj.unsqueeze(0)
            if intrinsics.dim() == 3:
                intr = intrinsics.unsqueeze(0)
            else:
                intr = intrinsics
            if imgs.dim() == 4:
                h, w = imgs.shape[-2:]
                sizes = torch.tensor([h, w], dtype=torch.float32, device=imgs.device)
                sizes = sizes.view(1, 1, 2).repeat(proj.shape[0], proj.shape[1], 1)
            else:
                raise ValueError("imgs must have shape (B*num_views, C, H, W) when passing raw tensors")
        return proj, intr, sizes

    def forward(
        self,
        mlvl_feats: Sequence[torch.Tensor],
        proj_mats: ProjectionBatch | ProjectionSample | torch.Tensor,
        imgs: torch.Tensor | None = None,
        world_points: torch.Tensor | None = None,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        distortion_coeff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del world_points, extrinsics, distortion_coeff  # not used yet
        proj_tensor, intr, image_sizes = self._prepare_projection(proj_mats, intrinsics, imgs)
        B, V, _, _ = proj_tensor.shape
        device = proj_tensor.device
        depth, height, width = self.voxel_shape
        world = self.world_grid.to(device)
        num_points = depth * height * width
        world_flat = world.view(1, num_points, 4).expand(B, num_points, 4)
        world_cols = world_flat.transpose(1, 2).unsqueeze(1)  # (B, 1, 4, N)
        cam_points = torch.matmul(proj_tensor, world_cols)
        cam_points = cam_points.transpose(-1, -2).view(B, V, depth, height, width, 4)

        depth_vals = cam_points[..., 2:3].clamp_min(1e-4)
        xy = cam_points[..., :2] / depth_vals

        sizes = image_sizes.to(device).view(B, V, 1, 1, 1, 2)
        width_px = sizes[..., 1:2].clamp_min(1.0)
        height_px = sizes[..., :1].clamp_min(1.0)
        x_norm = (xy[..., :1] / (width_px - 1).clamp_min(1.0)) * 2 - 1
        y_norm = (xy[..., 1:2] / (height_px - 1).clamp_min(1.0)) * 2 - 1
        grid = torch.cat([x_norm, y_norm], dim=-1)

        valid = (
            (xy[..., :1] >= 0)
            & (xy[..., :1] <= (width_px - 1))
            & (xy[..., 1:2] >= 0)
            & (xy[..., 1:2] <= (height_px - 1))
        )
        valid = valid.squeeze(-1) & (depth_vals.squeeze(-1) > 0)

        bev_levels = []
        grid_sample = grid.view(B * V, depth, height * width, 2)
        valid_view = valid.view(B, V, depth, height, width).unsqueeze(2)
        for feat in mlvl_feats:
            if feat.dim() != 4:
                raise ValueError("Feature maps must be 4D (N, C, H, W)")
            feat = feat.to(device)
            if feat.shape[0] != B * V:
                raise ValueError(
                    f"Feature batch ({feat.shape[0]}) does not match B*num_views ({B * V})"
                )
            sampled = F.grid_sample(
                feat,
                grid_sample,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled = sampled.view(B, V, feat.shape[1], depth, height, width)
            sampled = sampled * valid_view
            bev_levels.append(sampled.mean(dim=1))

        bev_feats = torch.stack(bev_levels, dim=0).mean(dim=0)
        world_points = world.view(1, depth, height, width, 4).to(device).expand(B, depth, height, width, 4)
        return bev_feats, world_points, cam_points


class DeformablePointSample(PointSample):
    """Lightweight implementation that projects multi-view features into BEV space."""

    def __init__(
        self,
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        num_cams: int,
        embed_dims: int,
        transformer: nn.Module | None = None,
        order: str = "NCDHW",
        depth_to_channels: bool = False,
        use_3d_reference: bool = False,
        skip_connection: bool = True,
        is_trace: bool = False,
        num_levels: int = 4,
        anchor_generator: Anchor3DRangeGenerator | Mapping | None = None,
        n_voxels: Sequence[int] | None = None,
        div: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            num_cams=num_cams,
            embed_dims=embed_dims,
            order=order,
            is_trace=is_trace,
            anchor_generator=anchor_generator,
            n_voxels=n_voxels,
            div=div,
        )
        self.transformer = transformer
        self.order2 = order
        self.depth_to_channels = depth_to_channels
        self.use_3d_reference = use_3d_reference
        self.skip_connection = skip_connection
        self.num_levels = num_levels
        if kwargs:
            # kwargs are accepted for API compatibility but unused
            pass

    def forward(
        self,
        mlvl_feats: Sequence[torch.Tensor],
        proj_mats: ProjectionBatch | ProjectionSample | torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        distortion_coeff: torch.Tensor | None = None,
        imgs: torch.Tensor | None = None,
        world_points: torch.Tensor | None = None,
        hidden_state_list: Sequence[torch.Tensor] | torch.Tensor | None = None,
        valid_bev_mask: torch.Tensor | None = None,
    ):
        del extrinsics, distortion_coeff, hidden_state_list
        mlvl_feats = list(mlvl_feats)[: self.num_levels]
        bev_feats, world_points, camera_points = super().forward(
            mlvl_feats,
            proj_mats,
            imgs=imgs,
            intrinsics=intrinsics,
        )

        if valid_bev_mask is not None:
            if valid_bev_mask.dim() == 4:
                valid_bev_mask = valid_bev_mask[:, None]
            bev_feats = bev_feats * valid_bev_mask

        order = self.order2
        if order == "NCDHW":
            pass
        elif order == "NDHWC":
            bev_feats = bev_feats.permute(0, 2, 3, 4, 1)
        elif order == "NCHWD":
            bev_feats = bev_feats.permute(0, 1, 3, 4, 2)
        elif order == "NDCHW":
            bev_feats = bev_feats.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(f"Unsupported order '{order}'")

        if self.depth_to_channels:
            if order not in {"NCDHW", "NDCHW"}:
                raise ValueError("depth_to_channels=True requires order 'NCDHW' or 'NDCHW'")
            shape = bev_feats.shape
            bev_feats = bev_feats.contiguous().view(shape[0], -1, shape[-2], shape[-1])

        return bev_feats.contiguous(), world_points, camera_points


MODULES.register("PointSample")(PointSample)
MODULES.register("DeformablePointSample")(DeformablePointSample)


__all__ = ["PointSample", "DeformablePointSample"]
