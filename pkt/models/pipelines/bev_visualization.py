"""Composable model that runs LightBackbone+FPN+Sampler for visualization."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

from pkt.data.nuscenes import ProjectionBatch, ProjectionSample
from pkt.engine.registries import MODULES, VISUALIZERS
from pkt.utils.build import build_from_cfg


def _projection_to_batch(projection, device: torch.device) -> ProjectionBatch:
    if isinstance(projection, ProjectionBatch):
        return projection.to(device)
    if isinstance(projection, ProjectionSample):
        return ProjectionBatch.from_samples([projection]).to(device)
    if isinstance(projection, list) and projection and isinstance(projection[0], ProjectionSample):
        return ProjectionBatch.from_samples(projection).to(device)
    raise TypeError("Unsupported projection type for BEVVisualizationModel")


@MODULES.register("BEVVisualizationModel")
class BEVVisualizationModel(nn.Module):
    def __init__(
        self,
        backbone: Dict[str, Any],
        neck: Dict[str, Any],
        view_selector: Dict[str, Any],
        sampler: Dict[str, Any],
        visualizers: Sequence[Dict[str, Any]] | None = None,
        cameras: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = build_from_cfg(backbone, MODULES)
        self.neck = build_from_cfg(neck, MODULES)
        self.view_selector = build_from_cfg(view_selector, MODULES)
        self.sampler = build_from_cfg(sampler, MODULES)
        self.visualizers = [build_from_cfg(cfg, VISUALIZERS) for cfg in (visualizers or [])]
        self.cameras = list(cameras) if cameras is not None else None

    def forward(self, inputs: Dict[str, Any]):
        images_dict = inputs["images"]
        selected = self.cameras or sorted(images_dict.keys())
        processed_images: List[torch.Tensor] = []
        for cam in selected:
            img = images_dict[cam]
            if img.dim() == 4:
                if img.shape[0] != 1:
                    raise ValueError("BEVVisualizationModel currently supports batch_size=1 for visualization")
                img = img.squeeze(0)
            if img.dim() != 3:
                raise ValueError(f"Expected image tensor with shape (C,H,W); got {tuple(img.shape)} for {cam}")
            processed_images.append(img)

        if len(processed_images) == 1:
            model_input = processed_images[0].unsqueeze(0)
        else:
            model_input = [img.unsqueeze(0) for img in processed_images]

        feats = self.backbone(model_input)
        feats = self.neck(feats)
        feats = self.view_selector(feats)

        projection = inputs.get("projection")
        if projection is None:
            raise KeyError("Inputs must contain 'projection' for BEVVisualizationModel")
        device = next(self.parameters()).device
        projection_batch = _projection_to_batch(projection, device)
        imgs_tensor = torch.stack(processed_images, dim=0).to(device)
        mlvl_feats = [feat.to(device) for feat in feats]
        bev_feats, world_pts, cam_pts = self.sampler(
            mlvl_feats,
            projection_batch,
            intrinsics=projection_batch.intrinsics,
            imgs=imgs_tensor,
        )

        result = {
            "feature_maps": list(zip(getattr(self.backbone, "out_strides", []), mlvl_feats)),
            "bev_feats": bev_feats,
            "images": [(name, img) for name, img in zip(selected, processed_images)],
            "selected_cams": selected,
            "world_points": world_pts,
            "camera_points": cam_pts,
        }
        # Inline visualizers (optional)
        for visualizer in self.visualizers:
            visualizer(result)
        return result


__all__ = ["BEVVisualizationModel"]
