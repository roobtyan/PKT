"""Multi-branch routing backbone that mirrors the production architecture with synthetic blocks."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from pkt.models.backbones import BACKBONE_REGISTRY


def _conv_block(in_channels: int, out_channels: int, *, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class MultiScaleEncoder(nn.Module):
    """Light-weight multi-scale encoder producing five resolution levels."""

    def __init__(self, in_channels: int, channels: Tuple[int, int, int, int, int]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        current = in_channels
        for idx, out_channels in enumerate(channels):
            stride = 1 if idx == 0 else 2
            self.blocks.append(_conv_block(current, out_channels, stride=stride))
            current = out_channels
        self.keys = ("x", "x1", "x2", "x3", "x4")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for key, block in zip(self.keys, self.blocks):
            x = block(x)
            outputs[key] = x
        return outputs


class MultiScaleEncoderSlim(nn.Module):
    """Slimmed-down encoder for the short-term distance branch."""

    def __init__(self, in_channels: int, channels: Tuple[int, int, int, int]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        current = in_channels
        for idx, out_channels in enumerate(channels):
            stride = 1 if idx == 0 else 2
            self.blocks.append(_conv_block(current, out_channels, stride=stride))
            current = out_channels
        self.keys = ("x_sd", "x1_sd", "x2_sd", "x3_sd")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for key, block in zip(self.keys, self.blocks):
            x = block(x)
            outputs[key] = x
        outputs["x4_sd"] = x
        return outputs


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = _conv_block(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MultiScaleEncoderUp(nn.Module):
    """UNet style up-sampling stack producing four upsampled feature maps."""

    def __init__(
        self,
        channels: Tuple[int, int, int, int, int],
        out_channels: Tuple[int, int, int, int],
    ) -> None:
        super().__init__()
        c4, c3, c2, c1, c0 = channels
        o4, o3, o2, o1 = out_channels
        self.up4 = UpBlock(c4, c3, o4)
        self.up3 = UpBlock(o4, c2, o3)
        self.up2 = UpBlock(o3, c1, o2)
        self.up1 = UpBlock(o2, c0, o1)

    def forward(
        self,
        x4: torch.Tensor,
        x3: torch.Tensor,
        x2: torch.Tensor,
        x1: torch.Tensor,
        x0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        up4 = self.up4(x4, x3)
        up3 = self.up3(up4, x2)
        up2 = self.up2(up3, x1)
        up1 = self.up1(up2, x0)
        return up1, up2, up3, up4


class AgentMapFusionV2(nn.Module):
    """Fuses agent embeddings with raster features to produce downstream context."""

    def __init__(self, map_channels: int, agent_channels: int, num_agents: int) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.proj = nn.Sequential(
            nn.Linear(map_channels + agent_channels + 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.score_head = nn.Linear(256, num_agents)

    def forward(
        self,
        map_sequence: torch.Tensor,
        agent_feature: torch.Tensor,
        crop_center: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        map_context = map_sequence.mean(dim=1)
        agent_context = agent_feature.mean(dim=1)
        crop_context = crop_center.mean(dim=1)
        fused = self.proj(torch.cat([map_context, agent_context, crop_context], dim=1))
        score = self.score_head(fused)
        return fused, score


class FeatureRefinementModule(nn.Module):
    """Produces trajectory offsets and selection scores from fused features."""

    def __init__(
        self,
        up_channels: Iterable[int],
        candidate_size: int,
        agent_dim: int,
        ego_dim: int,
        lane_dim: int,
        routing_modalities: int,
        num_points: int,
    ) -> None:
        super().__init__()
        pooled_dim = sum(up_channels)
        input_dim = pooled_dim + candidate_size + agent_dim + ego_dim + lane_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.offset_head = nn.Linear(256, routing_modalities * num_points * 2)
        self.score_head = nn.Linear(256, routing_modalities)
        self.routing_modalities = routing_modalities
        self.num_points = num_points

    def forward(
        self,
        up_features: Tuple[torch.Tensor, ...],
        candidate: torch.Tensor,
        agent_feature: torch.Tensor,
        ego_feature: torch.Tensor,
        lane_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = [F.adaptive_avg_pool2d(feat, 1).flatten(1) for feat in up_features]
        candidate_flat = candidate.reshape(candidate.size(0), -1)
        agent_context = agent_feature.mean(dim=1)
        fused = torch.cat(pooled + [candidate_flat, agent_context, ego_feature, lane_feature], dim=1)
        hidden = self.mlp(fused)
        offset = self.offset_head(hidden).view(candidate.size(0), self.routing_modalities, self.num_points, 2)
        score = self.score_head(hidden)
        return offset, score


class ObjectDecisionDecoder(nn.Module):
    """Generates interaction probabilities for downstream decision modules."""

    def __init__(self, feature_dim: int, num_agents: int) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.odm_head = nn.Linear(128, num_agents)
        self.abn_head = nn.Linear(128, num_agents)

    def forward(
        self,
        feature: torch.Tensor,
        crop_index: torch.Tensor,
        map_info: torch.Tensor,
        agent_speed: torch.Tensor,
        agent_mask: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        crop_flat = crop_index.reshape(crop_index.size(0), -1)
        traj_flat = trajectory.reshape(trajectory.size(0), -1)
        fused = torch.cat([pooled, crop_flat, map_info, agent_speed, agent_mask, traj_flat], dim=1)
        hidden = self.mlp(fused)
        return torch.sigmoid(self.odm_head(hidden)), torch.sigmoid(self.abn_head(hidden))


@BACKBONE_REGISTRY.register("routing_perception")
class RoutingPerceptionBackbone(nn.Module):
    """Pseudo routing backbone that mirrors the multi-branch production layout."""

    def __init__(
        self,
        *,
        raster_channels: int = 28,
        sd_channels: int = 2,
        raster_size: Tuple[int, int] = (64, 64),
        num_points: int = 100,
        routing_modalities: int = 5,
        motion_modalities: int = 3,
        lon_modalities: int = 5,
        lon_steps: int = 50,
        agent_feature_dim: int = 76,
        num_agents: int = 8,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.routing_modalities = routing_modalities
        self.motion_modalities = motion_modalities
        self.lon_modalities = lon_modalities
        self.lon_steps = lon_steps
        self.num_agents = num_agents

        env_channels = (32, 64, 96, 128, 160)
        sd_branch_channels = (16, 32, 48, 64)
        up_channels = (160, 128, 96, 64, 32)
        up_out_channels = (128, 96, 64, 48)
        self.env_encoder = MultiScaleEncoder(raster_channels, env_channels)
        self.sd_encoder = MultiScaleEncoderSlim(sd_channels, sd_branch_channels)
        self.sd_align3 = nn.Conv2d(sd_branch_channels[-1], env_channels[-2], kernel_size=1)
        self.sd_align4 = nn.Conv2d(sd_branch_channels[-1], env_channels[-1], kernel_size=1)
        self.unet = MultiScaleEncoderUp(up_channels, up_out_channels)

        h, w = raster_size
        x4_hw = (h // 16, w // 16)
        sd_hw = (h // 8, w // 8)
        self.flat_x4_dim = env_channels[-1] * x4_hw[0] * x4_hw[1]
        self.flat_sd_dim = sd_branch_channels[-1] * sd_hw[0] * sd_hw[1]

        decode_dim = 256
        self.dec_fc = nn.Sequential(
            nn.Linear(self.flat_x4_dim, decode_dim),
            nn.ReLU(inplace=True),
        )
        self.sd_fc = nn.Sequential(
            nn.Linear(self.flat_sd_dim, decode_dim),
            nn.ReLU(inplace=True),
        )
        self.decode_points_fc = nn.Linear(
            decode_dim * 2, (routing_modalities + 1) * num_points * 2
        )

        lcc_hidden = 256
        self.lcc_fc = nn.Sequential(
            nn.Linear(self.flat_x4_dim, lcc_hidden),
            nn.ReLU(inplace=True),
        )
        self.lcc_reg = nn.Linear(lcc_hidden, motion_modalities * num_points * 2)
        self.lcc_cls = nn.Linear(lcc_hidden, motion_modalities)

        self.x4_embedding = nn.Conv2d(env_channels[-1], 128, kernel_size=1)
        self.agent_encoder = nn.Linear(agent_feature_dim, 256)
        self.map_fusion = AgentMapFusionV2(map_channels=128, agent_channels=256, num_agents=num_agents)

        self.ego_encoder = nn.Sequential(
            nn.Linear(32 + 16 + 48, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
        )

        self.bitmap_embed = nn.Linear(32, 32)
        self.arrow_embed = nn.Linear(16, 32)
        self.main_action_embed = nn.Linear(16, 32)
        self.assist_action_embed = nn.Linear(16, 32)
        self.distance_embed = nn.Linear(8, 32)
        self.lane_fusion = nn.Sequential(
            nn.Linear(32 * 5, 128),
            nn.ReLU(inplace=True),
        )

        candidate_size = routing_modalities * num_points * 2
        refine_up_channels = tuple(reversed(up_out_channels))
        self.refine_module = FeatureRefinementModule(
            up_channels=refine_up_channels,
            candidate_size=candidate_size,
            agent_dim=256,
            ego_dim=32,
            lane_dim=128,
            routing_modalities=routing_modalities,
            num_points=num_points,
        )

        self.extra_conv = _conv_block(up_out_channels[1], up_out_channels[1])
        odm_feature_dim = (
            up_out_channels[1]
            + num_agents * (4 + 2)
            + 48
            + (routing_modalities + motion_modalities) * num_points * 2
        )
        self.odm_decoder = ObjectDecisionDecoder(odm_feature_dim, num_agents=num_agents)

        lon_input_dim = decode_dim + 32
        self.lon_proj = nn.Linear(lon_input_dim, lon_modalities * lon_steps)
        self.lon_score = nn.Linear(lon_input_dim, lon_modalities)
        self.lon_base = nn.Linear(lon_input_dim, lon_steps)
        self.ego_speed_proj = nn.Linear(lon_input_dim, lon_modalities * lon_steps)

        self.output_dim = decode_dim

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        raster = inputs["backbone_img_feature"]
        sd = inputs["short_sd_feature"]
        env_feats = self.env_encoder(raster)
        sd_feats = self.sd_encoder(sd)

        sd_x4 = F.interpolate(
            self.sd_align4(sd_feats["x4_sd"]),
            size=env_feats["x4"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        sd_x3 = F.interpolate(
            self.sd_align3(sd_feats["x3_sd"]),
            size=env_feats["x3"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x4 = env_feats["x4"] + sd_x4
        x3_sum = env_feats["x3"] + sd_x3
        up_x_1, up_x_2, up_x_3, up_x_4 = self.unet(
            x4,
            x3_sum,
            env_feats["x2"],
            env_feats["x1"],
            env_feats["x"],
        )

        decoded_feat = self.dec_fc(x4.flatten(1))
        sd_feat = self.sd_fc(sd_feats["x4_sd"].flatten(1))
        decoded_points = self.decode_points_fc(torch.cat([decoded_feat, sd_feat], dim=1))
        raw_trajectory = decoded_points.view(
            raster.size(0), self.routing_modalities + 1, self.num_points, 2
        )
        zero_trajectory = raw_trajectory[:, 0]
        candidate_traj = raw_trajectory[:, 1:]
        detached_candidate = candidate_traj.detach()

        lcc_feat = self.lcc_fc(x4.flatten(1))
        motion_points = self.lcc_reg(lcc_feat).view(
            raster.size(0), self.motion_modalities, self.num_points, 2
        )
        motion_score = self.lcc_cls(lcc_feat)

        x4_emb = self.x4_embedding(x4)
        x4_seq = x4_emb.flatten(2).transpose(1, 2)
        agent_feature = self.agent_encoder(inputs["agent_feature"])
        fusion_feature, map_score = self.map_fusion(x4_seq, agent_feature, inputs["crop_center"])

        ego_inputs = torch.cat(
            [inputs["ego_info"], inputs["traffic_light_feature"], inputs["map_info"]], dim=1
        )
        ego_feature = self.ego_encoder(ego_inputs)

        lane_components = torch.cat(
            [
                self.bitmap_embed(inputs["bitmap_feature"]),
                self.arrow_embed(inputs["arrow_feature"]),
                self.main_action_embed(inputs["main_action_feature"]),
                self.assist_action_embed(inputs["assistant_action_feature"]),
                self.distance_embed(inputs["distance_feature"]),
            ],
            dim=1,
        )
        lane_feature = self.lane_fusion(lane_components)

        offset, score_refine = self.refine_module(
            (up_x_1, up_x_2, up_x_3, up_x_4),
            detached_candidate,
            agent_feature,
            ego_feature,
            lane_feature,
        )
        refined_traj = candidate_traj + offset

        routing_traj = torch.cat([refined_traj, motion_points], dim=1)
        score = torch.cat([score_refine, motion_score], dim=1)

        lon_input = torch.cat([decoded_feat, ego_feature], dim=1)
        lon = self.lon_proj(lon_input).view(raster.size(0), self.lon_modalities, self.lon_steps)
        lon_score = self.lon_score(lon_input)
        lon_base = self.lon_base(lon_input)
        ego_speed_pred = self.ego_speed_proj(lon_input).view(
            raster.size(0), self.lon_modalities, self.lon_steps
        )

        extra_feature = self.extra_conv(up_x_3)
        odm_prob, abn_prob = self.odm_decoder(
            extra_feature,
            inputs["crop_index_feature"],
            inputs["map_info"],
            inputs["agent_cur_speed"],
            inputs["agent_mask"],
            routing_traj,
        )

        return {
            "logits": score,
            "routing_trajectory": routing_traj,
            "raw_trajectory": raw_trajectory,
            "score": score,
            "lcc_score": motion_score,
            "lcc_trajectories": motion_points,
            "lon": lon,
            "lon_score": lon_score,
            "lon_base": lon_base,
            "ego_speed": ego_speed_pred,
            "fusion_feature": fusion_feature,
            "map_score": map_score,
            "odm_prob": odm_prob,
            "abn_static_prob": abn_prob,
            "zero_trajectory": zero_trajectory,
            "refined_trajectories": refined_traj,
        }
