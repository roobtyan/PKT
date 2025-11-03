"""Routing related objectives used for synthetic training."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


def rotate_refline(refline_points: torch.Tensor, start_idx: int, angle: torch.Tensor) -> torch.Tensor:
    """Rotate a reference line around ``start_idx`` by ``angle`` radians."""
    angles = angle.view(-1, 1, 1)
    tail = refline_points[:, start_idx:]
    origin = tail[:, :1]
    centered = tail - origin

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    x_new = centered[..., 0:1] * cos_a - centered[..., 1:2] * sin_a
    y_new = centered[..., 0:1] * sin_a + centered[..., 1:2] * cos_a
    rotated_tail = torch.cat([x_new, y_new], dim=-1) + origin

    return torch.cat([refline_points[:, :start_idx], rotated_tail], dim=1)


def get_speed_range_weight(cur_ego_speed: torch.Tensor, preview_time: float) -> torch.Tensor:
    """Weight trajectory points based on current ego speed."""
    bs = cur_ego_speed.shape[0]
    device = cur_ego_speed.device
    preview_dist = torch.clamp(preview_time * cur_ego_speed.unsqueeze(1) + 5.0, 10.0, 80.0).int()

    weights = torch.ones(bs, 80, device=device) * 0.5
    for idx in range(bs):
        limit = int(preview_dist[idx])
        ramp = torch.linspace(15.0, 5.0, limit, device=device)
        weights[idx, :limit] = ramp
    head = torch.ones(bs, 20, device=device) * 0.1
    return torch.cat([head, weights], dim=1)


def build_anchor(zero_path: torch.Tensor, cur_ego_speed: torch.Tensor, num_modalities: int) -> torch.Tensor:
    """Create anchor trajectories by shifting the reference path laterally."""
    bs, num_points, _ = zero_path.shape
    base = zero_path.unsqueeze(1).repeat(1, num_modalities, 1, 1)
    offsets = torch.linspace(-1.0, 1.0, steps=num_modalities, device=zero_path.device).view(1, -1, 1)
    scale = 0.1 + cur_ego_speed.view(-1, 1, 1) * 0.02
    shift = offsets.expand(bs, -1, num_points) * scale
    base[..., 1] = base[..., 1] + shift
    return base


def compute_lateral_dist(gt_pos: torch.Tensor, pd_pos: torch.Tensor) -> torch.Tensor:
    """Compute simple squared distance for lateral error approximation."""
    diff = gt_pos.unsqueeze(2) - pd_pos
    return torch.sum(diff[..., 1] ** 2, dim=-1)


class RoutingTrajectoryObjective(nn.Module):
    """Supervise refined and raw routing trajectories."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = weight
        self.l1_loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, *, logits: torch.Tensor, target: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor], **_: object) -> torch.Tensor:
        gt_points = target["structure:agents_trajectory_trace"]
        gt_mask = target["structure:agents_trajectory_trace_mask"].unsqueeze(-1)
        gt_mask_ratio = gt_mask.to(torch.float32).mean(dim=1).squeeze(1) + 1e-4

        pred_refine_full = output["routing_trajectory"]
        raw_bundle = output["raw_trajectory"]
        candidate_count = raw_bundle.shape[1] - 1
        pred_refine = pred_refine_full[:, :candidate_count]
        zero_path = raw_bundle[:, 0]
        raw_paths = raw_bundle[:, 1:]

        cur_speed = target["structure:cur_ego_speed"]
        anchor = build_anchor(zero_path, cur_speed, candidate_count)

        refline_weight = target["structure:refline_sample_weight"]

        distance = (gt_points.unsqueeze(1) - anchor) ** 2
        distance = distance * gt_mask.unsqueeze(1)
        speed_weight = get_speed_range_weight(cur_speed, preview_time=5.0)
        modal_weight = speed_weight.unsqueeze(1).repeat(1, anchor.shape[1], 1)
        distance = torch.sum(distance, dim=-1) * modal_weight
        distance = torch.sum(distance, dim=-1)
        idx = torch.argmin(distance, dim=-1)

        mask_gt = gt_points * gt_mask
        pred_ref = pred_refine[torch.arange(gt_points.size(0), device=gt_points.device), idx]
        pred_ref = pred_ref * gt_mask

        pred_raw = raw_paths[torch.arange(gt_points.size(0), device=gt_points.device), idx]
        pred_raw = pred_raw * gt_mask

        pos_refine = self.l1_loss(mask_gt, pred_ref)
        pos_raw = self.l1_loss(mask_gt, pred_raw)
        loss_long = pos_refine + pos_raw

        loss_long = loss_long.mean(dim=-1) * speed_weight
        loss_long = torch.mean(loss_long, dim=1) / gt_mask_ratio

        ref_path_loss = self.l1_loss(mask_gt, zero_path * gt_mask).mean(dim=(1, 2)) / gt_mask_ratio

        spacing = torch.norm(torch.diff(pred_refine_full, dim=2), dim=-1)
        spacing_loss = F.mse_loss(spacing, torch.ones_like(spacing), reduction="none").sum(dim=1).mean()

        loss = torch.sum(loss_long * refline_weight) / ((refline_weight != 0).sum() + 1e-6)
        zero_loss = torch.sum(ref_path_loss * refline_weight) / ((refline_weight != 0).sum() + 1e-6)

        motion_pred = output["lcc_trajectories"]
        motion_distance = (gt_points.unsqueeze(1) - motion_pred) ** 2
        motion_distance = motion_distance * gt_mask.unsqueeze(1)
        motion_distance = torch.sum(motion_distance, dim=-1) * speed_weight.unsqueeze(1).repeat(1, motion_pred.size(1), 1)
        motion_distance = torch.sum(motion_distance, dim=-1)
        motion_idx = torch.argmin(motion_distance, dim=-1)
        chosen_motion = motion_pred[torch.arange(gt_points.size(0), device=gt_points.device), motion_idx]
        motion_loss = self.l1_loss(mask_gt, chosen_motion * gt_mask)
        motion_loss = motion_loss.mean(dim=-1) * speed_weight
        motion_loss = torch.mean(motion_loss, dim=1)
        motion_loss = torch.sum(motion_loss * refline_weight) / ((refline_weight != 0).sum() + 1e-6)

        return self.loss_weight * (loss + 0.1 * spacing_loss + 20.0 * zero_loss + motion_loss)


class TrajectoryClassificationObjective(nn.Module):
    """Classification loss aligning discrete trajectories with ground truth."""

    def __init__(self, weight: float = 1.0, lcc_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = weight
        self.lcc_weight = lcc_weight

    def forward(self, *, logits: torch.Tensor, target: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor], **_: object) -> torch.Tensor:
        gt_points = target["structure:agents_trajectory_trace"]
        gt_mask = target["structure:agents_trajectory_trace_mask"].unsqueeze(-1)
        cur_speed = target["structure:cur_ego_speed"]
        refline_weight = target["structure:refline_sample_weight"]

        pred_traj = output["routing_trajectory"]
        speed_weight = get_speed_range_weight(cur_speed, 5.0).unsqueeze(1).repeat(1, pred_traj.size(1), 1)

        distance = (gt_points.unsqueeze(1) - pred_traj) ** 2
        distance = distance * gt_mask.unsqueeze(1)
        distance = distance.mean(dim=-1) * speed_weight
        distance = torch.sum(distance, dim=-1)
        idx = torch.argmin(distance, dim=-1)

        score = output["score"]
        ce_loss = F.cross_entropy(score, idx.long(), reduction="none")
        ce_loss = torch.sum(ce_loss * refline_weight) / ((refline_weight != 0).sum() + 1e-6)

        lcc_traj = output["lcc_trajectories"]
        lcc_distance = (gt_points.unsqueeze(1) - lcc_traj) ** 2
        lcc_distance = lcc_distance * gt_mask.unsqueeze(1)
        lcc_distance = lcc_distance.mean(dim=-1) * speed_weight[:, :lcc_traj.size(1)]
        lcc_distance = torch.sum(lcc_distance, dim=-1)
        lcc_idx = torch.argmin(lcc_distance, dim=-1)

        lcc_score = output["lcc_score"]
        lcc_ce = F.cross_entropy(lcc_score, lcc_idx.long(), reduction="none")
        lcc_ce = torch.sum(lcc_ce * refline_weight) / ((refline_weight != 0).sum() + 1e-6)

        return self.loss_weight * ce_loss + self.lcc_weight * lcc_ce


class STLonObjective(nn.Module):
    """Loss on longitudinal trajectories and ego speed predictions."""

    def __init__(self, weight: float = 1.0, lon_base_only: bool = False) -> None:
        super().__init__()
        self.weight = weight
        self.lon_base_only = lon_base_only
        self.l1 = nn.SmoothL1Loss(reduction="none")

    def forward(self, *, logits: torch.Tensor, target: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor], **_: object) -> torch.Tensor:
        refline_weight = target["structure:refline_sample_weight"]
        lon = output["lon"]
        lon_score = output["lon_score"]
        lon_base = output["lon_base"]
        ego_speed_pred = output["ego_speed"]

        bs, lon_modal, pred_len = lon.shape

        a_small = 0.5
        a_mid = 1.0
        delta_small = 0.5 * a_small * (torch.arange(pred_len, device=lon.device) * 0.1) ** 2
        delta_mid = 0.5 * a_mid * (torch.arange(pred_len, device=lon.device) * 0.2) ** 2

        lon_anchor = torch.stack(
            (
                lon[:, 0],
                lon[:, 0] + delta_small.clamp(max=0.5),
                lon[:, 0] - delta_small.clamp(max=0.5),
                lon[:, 0] + delta_mid.clamp(max=2.0),
                lon[:, 0] - delta_mid.clamp(max=2.0),
            ),
            dim=1,
        )

        ego_info = target["targets:future_ego_agent_info"][:, 0, : pred_len + 1]
        ego_delta = torch.norm(ego_info[ :, 1:, 2:4] - ego_info[:, :-1, 2:4], dim=-1)
        ego_s = torch.cumsum(ego_delta, dim=1)

        time_mask = ego_info[:, :, 16].bool()
        valid_mask = time_mask.all(dim=1) & (~target["bad_sample"]) & target["agent_trace_all_valid"]
        valid_mask = valid_mask.float()

        lon_dist = torch.abs(lon_anchor - ego_s.unsqueeze(1)).mean(dim=-1)
        lon_idx = torch.argmin(lon_dist, dim=1)

        lon_score_loss = F.cross_entropy(lon_score, lon_idx.long(), reduction="none")
        lon_selected = lon[torch.arange(bs, device=lon.device), lon_idx]
        lon_loss = self.l1(lon_selected, ego_s).mean(dim=-1)

        lon_base_loss = self.l1(lon_base, ego_s).mean(dim=-1)

        ego_speed = target["targets:future_ego_speed"][:, :pred_len]
        speed_mask = target["targets:ego_speed_mask"][:, :pred_len]
        selected_speed = ego_speed_pred[torch.arange(bs, device=lon.device), lon_idx]
        speed_loss = self.l1(selected_speed, ego_speed).mean(dim=-1)

        weight_sum = (refline_weight != 0).sum() + 1e-6
        lon_score_loss = torch.sum(lon_score_loss * refline_weight * valid_mask) / weight_sum
        lon_loss = torch.sum(lon_loss * refline_weight * valid_mask) / weight_sum
        lon_base_loss = torch.sum(lon_base_loss * refline_weight * valid_mask) / weight_sum
        speed_valid = speed_mask.all(dim=1) & (~target["bad_sample"])
        speed_valid = speed_valid.float()
        speed_loss = torch.sum(speed_loss * refline_weight * speed_valid) / weight_sum

        if self.lon_base_only:
            return self.weight * lon_base_loss

        return self.weight * (lon_loss + speed_loss + lon_score_loss + lon_base_loss)


__all__ = [
    "RoutingTrajectoryObjective",
    "TrajectoryClassificationObjective",
    "STLonObjective",
]
