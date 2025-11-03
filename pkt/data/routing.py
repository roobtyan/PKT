"""Utility dataset producing synthetic routing supervision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from pkt.data.datasets import BaseDataset, DATASET_REGISTRY


@dataclass(frozen=True)
class RoutingSampleConfig:
    """Shape configuration for the synthetic routing dataset."""

    num_points: int = 100
    preview_points: int = 50
    routing_modalities: int = 5
    lcc_modalities: int = 3
    lon_modalities: int = 5
    lon_steps: int = 50
    raster_shape: Tuple[int, int, int] = (28, 64, 64)
    sd_shape: Tuple[int, int, int] = (2, 64, 64)
    num_agents: int = 8
    agent_feature_dim: int = 76
    crop_index_dim: int = 4
    ego_info_dim: int = 32
    traffic_light_dim: int = 16
    map_info_dim: int = 48
    bitmap_dim: int = 32
    arrow_dim: int = 16
    action_dim: int = 16
    distance_dim: int = 8


def _random_refline(generator: torch.Generator, num_points: int) -> torch.Tensor:
    steps = torch.randn(num_points, 2, generator=generator) * 0.5
    return torch.cumsum(steps, dim=0)


def _random_inputs(generator: torch.Generator, cfg: RoutingSampleConfig) -> Dict[str, torch.Tensor]:
    """Create placeholder inputs that mirror the production routing contract."""

    raster = torch.randn(cfg.raster_shape, generator=generator)
    sd = torch.randn(cfg.sd_shape, generator=generator)
    agent_feature = torch.randn(cfg.num_agents, cfg.agent_feature_dim, generator=generator)
    crop_center = torch.randn(cfg.num_agents, 2, generator=generator)
    crop_index = torch.randn(cfg.num_agents, cfg.crop_index_dim, generator=generator)
    ego_info = torch.randn(cfg.ego_info_dim, generator=generator)
    traffic_light = torch.randn(cfg.traffic_light_dim, generator=generator)
    map_info = torch.randn(cfg.map_info_dim, generator=generator)
    bitmap = torch.randn(cfg.bitmap_dim, generator=generator)
    arrow = torch.randn(cfg.arrow_dim, generator=generator)
    main_action = torch.randn(cfg.action_dim, generator=generator)
    assist_action = torch.randn(cfg.action_dim, generator=generator)
    distance = torch.randn(cfg.distance_dim, generator=generator)
    agent_speed = torch.abs(torch.randn(cfg.num_agents, generator=generator)) * 10.0
    agent_mask = torch.ones(cfg.num_agents, dtype=torch.float32)

    return {
        "backbone_img_feature": raster,
        "short_sd_feature": sd,
        "agent_feature": agent_feature,
        "crop_center": crop_center,
        "crop_index_feature": crop_index,
        "ego_info": ego_info,
        "traffic_light_feature": traffic_light,
        "map_info": map_info,
        "bitmap_feature": bitmap,
        "arrow_feature": arrow,
        "main_action_feature": main_action,
        "assistant_action_feature": assist_action,
        "distance_feature": distance,
        "agent_cur_speed": agent_speed,
        "agent_mask": agent_mask,
    }


def _random_targets(generator: torch.Generator, cfg: RoutingSampleConfig) -> Dict[str, torch.Tensor]:
    num_points = cfg.num_points
    preview_points = cfg.preview_points

    gt_refline = _random_refline(generator, num_points)
    mask = torch.ones(num_points, dtype=torch.float32)

    cur_speed = torch.rand((), generator=generator) * 30.0
    refline_weight = torch.ones((), dtype=torch.float32)

    ego_positions = torch.zeros(preview_points + 1, 20, dtype=torch.float32)
    displacements = torch.randn(preview_points, 2, generator=generator) * 0.2
    ego_positions[1:, 2:4] = torch.cumsum(displacements, dim=0)
    ego_positions[:, 16] = 1.0

    future_speed = torch.abs(torch.randn(preview_points, generator=generator)) * 5.0
    speed_mask = torch.ones(preview_points, dtype=torch.bool)

    return {
        "structure:agents_trajectory_trace": gt_refline,
        "structure:agents_trajectory_trace_mask": mask,
        "structure:cur_ego_speed": cur_speed,
        "structure:refline_sample_weight": refline_weight,
        "targets:future_ego_agent_info": ego_positions.unsqueeze(0),
        "targets:future_ego_speed": future_speed,
        "targets:ego_speed_mask": speed_mask,
        "bad_sample": torch.tensor(False, dtype=torch.bool),
        "agent_trace_all_valid": torch.tensor(True, dtype=torch.bool),
    }


@DATASET_REGISTRY.register("random_routing")
class RandomRoutingDataset(BaseDataset):
    """Synthetic dataset yielding structured routing supervision."""

    def __init__(
        self,
        num_samples: int,
        *,
        seed: int | None = None,
        sample_cfg: RoutingSampleConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = sample_cfg or RoutingSampleConfig()
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

        self.inputs: Tuple[Dict[str, torch.Tensor], ...] = tuple(
            _random_inputs(self.generator, self.cfg) for _ in range(num_samples)
        )
        self.targets: Tuple[Dict[str, torch.Tensor], ...] = tuple(
            _random_targets(self.generator, self.cfg) for _ in range(num_samples)
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:  # type: ignore[override]
        inputs = {k: v.clone() for k, v in self.inputs[index].items()}
        targets = {k: v.clone() for k, v in self.targets[index].items()}
        return inputs, targets
