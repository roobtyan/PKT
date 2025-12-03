#!/usr/bin/env python
"""
Visualize LightBackbone feature maps on a NuScenes sample.
Usage：
python pkt/visualize/visualize_light_backbone.py --dataroot data/v1.0-mini \
    --version v1.0-mini --split mini_train --sample-index 0 --cat-dim 0 \
    --channels 8 --cols 4 --output outputs/light_backbone_features.png  \
    --bev-output outputs/sample_bev.png  --point-cloud-range -40 -40 -4 140 40 2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pkt.data.nuscenes import NuScenesLidarFusionDataset, ProjectionBatch, fuse_projection
from pkt.models.light_backbone import LightBackbone
from pkt.models.point_sample import DeformablePointSample
from pkt.models.fpn import FPN, ViewSelector
from pkt.models.anchors import AlignedAnchor3DRangeGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LightBackbone on a NuScenes sample and plot feature maps.")
    parser.add_argument("--dataroot", type=str, required=True, help="Root directory of the NuScenes dataset.")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="NuScenes version string.")
    parser.add_argument(
        "--split",
        type=str,
        default="mini_train",
        help="Dataset split understood by nuscenes-devkit (train/val/mini_train/mini_val/etc.).",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset index to visualize.")
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help="Camera channels to use (defaults to all available in alphabetical order).",
    )
    parser.add_argument(
        "--cat-dim",
        type=int,
        default=1,
        help="Concatenate dimension when feeding multiple images (matches LightBackbone.cat_dim).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to run the backbone on.")
    parser.add_argument("--channels", type=int, default=8, help="How many channels to render per feature level.")
    parser.add_argument("--cols", type=int, default=4, help="How many columns per row in the feature grid.")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/light_backbone_features.png",
        help="Path to save the rendered figure.",
    )
    parser.add_argument(
        "--bev-output",
        type=str,
        default=None,
        help="Optional path to save a BEV heatmap rendered from fused features.",
    )
    parser.add_argument(
        "--bev-channel",
        type=int,
        default=0,
        help="Channel index (after depth reduction) to visualize in the BEV heatmap.",
    )
    parser.add_argument(
        "--bev-reduction",
        choices=["mean", "max"],
        default="mean",
        help="How to collapse the depth dimension before projecting onto BEV.",
    )
    parser.add_argument(
        "--bev-colormap",
        type=str,
        default="magma",
        help="Matplotlib colormap used for the BEV heatmap.",
    )
    parser.add_argument(
        "--point-cloud-range",
        type=float,
        nargs=6,
        default=(-30.0, -30.0, -4.0, 30.0, 30.0, 2.0),
        metavar=("x_min", "y_min", "z_min", "x_max", "y_max", "z_max"),
        help="Spatial bounds of the BEV grid (meters).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 2.0),
        metavar=("vx", "vy", "vz"),
        help="Voxel size (meters) along x/y/z for the BEV sampler.",
    )
    return parser.parse_args()


def normalize_feature(channel: torch.Tensor) -> torch.Tensor:
    """Min-max normalize a single feature channel for plotting."""

    c_min, c_max = channel.min(), channel.max()
    denom = (c_max - c_min).clamp(min=1e-6)
    return (channel - c_min) / denom


def plot_features(
        images: Sequence[Tuple[str, torch.Tensor]],
        feature_maps: Sequence[Tuple[int, torch.Tensor]],
        *,
        channels: int,
        cols: int,
        output: Path,
) -> None:
    """Create a grid figure. Supports (B, C, H, W) where B is num_views."""

    # 1. 确定有多少个 View
    # 假设 images 列表长度就是 View 数量
    num_views = len(images)

    # 2. 计算总行数
    # Image 行: 每个 View 一张图
    # Feature 行: 每个 Feature Level (Stride) 展示 'channels' 个通道
    # 为了清晰，建议布局改为：
    # Row 0: [Cam1] [Cam2] [Cam3] ...
    # Row 1 (Stride 8): [Cam1_Ch0] [Cam2_Ch0] [Cam3_Ch0] ... (这样能对比同一个通道在不同视角的表现)

    # 这里我们采用一种简单的平铺策略：
    # 每一组特征图，我们要画 (num_views * channels) 个子图

    rows_images = math.ceil(num_views / cols)

    rows_features = 0
    feature_meta = []  # Store (stride, num_channels_to_plot)

    for stride, fmap in feature_maps:
        # fmap shape: (N_views, C, H, W) 或者 (1, C, H, W)
        if fmap.dim() == 4:
            n, c, h, w = fmap.shape
        else:
            # 兼容 (C, H, W) 情况
            n, c, h, w = 1, fmap.shape[0], fmap.shape[1], fmap.shape[2]

        plot_c = min(channels, c)
        # 我们要画 n * plot_c 个图
        total_subplots = n * plot_c
        level_rows = math.ceil(total_subplots / cols)

        rows_features += level_rows
        feature_meta.append((stride, n, plot_c))

    rows_total = rows_images + rows_features
    # 适当增加图的高度以容纳内容
    fig = plt.figure(figsize=(cols * 3, rows_total * 2.5))
    grid = fig.add_gridspec(rows_total, cols)

    # --- Plot Original Images ---
    for idx, (name, img) in enumerate(images):
        r = idx // cols
        c = idx % cols
        ax_img = fig.add_subplot(grid[r, c])
        # img shape (C, H, W)
        img_np = img.permute(1, 2, 0).detach().cpu().clamp(0.0, 1.0).numpy()
        ax_img.imshow(img_np)
        ax_img.set_title(f"{name}", fontsize=8)
        ax_img.axis("off")

    # --- Plot Features ---
    row_cursor = rows_images

    for (stride, fmap), (stride_meta, n_views, n_channels) in zip(feature_maps, feature_meta):
        # fmap: (N, C, H, W)
        fmap = fmap.detach().cpu()
        if fmap.dim() == 3: fmap = fmap.unsqueeze(0)

        # 我们希望把同一个 Channel 的不同 View 放在一起对比，或者按 View 顺序排
        # 这里按：View 1 (all channels) -> View 2 (all channels) 顺序画

        cnt = 0
        for v in range(n_views):
            # 获取对应 View 的名称 (如果有的话)
            view_name = images[v][0] if v < len(images) else f"View{v}"

            for ch in range(n_channels):
                # 计算当前子图的位置
                r = row_cursor + cnt // cols
                c = cnt % cols

                ax = fig.add_subplot(grid[r, c])

                # 取出 (View v, Channel ch)
                feat_data = fmap[v, ch]  # (H, W)
                normalized = normalize_feature(feat_data).numpy()

                ax.imshow(normalized, cmap="viridis")  # 推荐 viridis 或 magma
                # 标题标明：S=Stride, V=View, C=Channel
                ax.set_title(f"S{stride}-{view_name}-C{ch}", fontsize=6)
                ax.axis("off")

                cnt += 1

        # 更新行游标
        row_cursor += math.ceil(cnt / cols)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] Saved feature visualization to {output}")


def main() -> None:
    base_out_stride = [8, 16, 32]
    backbone_out_channels = [96, 128, 256]
    embed_dims = 48
    stride_out = 8
    stride_out_channel = backbone_out_channels[base_out_stride.index(stride_out)]

    args = parse_args()
    dataroot = Path(args.dataroot).expanduser().resolve()
    output_path = Path(args.output).expanduser()

    dataset = NuScenesLidarFusionDataset(
        dataroot=str(dataroot),
        version=args.version,
        split=args.split,
        load_annotations=False,
        skip_empty=False,
    )
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} out of range (dataset has {len(dataset)} samples)")

    inputs, _ = dataset[args.sample_index]
    images: dict[str, torch.Tensor] = inputs["images"]
    available = sorted(images.keys())
    selected_cams = args.cameras or available
    missing = [c for c in selected_cams if c not in images]
    if missing:
        raise KeyError(f"Cameras not available: {missing}. Options: {available}")

    # LightBackbone supports sequence input; concatenate along cat_dim.
    image_list = [images[c] for c in selected_cams]
    reference_image = image_list[0]  # use first for channel inference

    device = torch.device(args.device)
    if args.cat_dim == 0:
        in_ch = reference_image.shape[0]
    else:
        # 只有 cat_dim=1 (channel concat) 时才需要累加通道
        in_ch = reference_image.shape[0] if len(image_list) == 1 else sum(img.shape[0] for img in image_list)

    backbone = LightBackbone(in_channels=in_ch, stem_channels=32, cat_dim=args.cat_dim, out_strides=base_out_stride,
                             stride_out=stride_out)
    fpn_backbone = FPN(in_channels=backbone_out_channels, out_channels=embed_dims, num_outs=4, sep_conv=False)
    view_selector = ViewSelector(list(range(len(selected_cams))), len(selected_cams))

    backbone.to(device).eval()

    with torch.no_grad():
        if len(image_list) == 1:
            model_input = reference_image.unsqueeze(0).to(device)  # (B, C, H, W)
        else:
            model_input = [img.unsqueeze(0).to(device) for img in image_list]  # sequence input
        # 视觉分支
        feats = backbone(model_input)
        feats = fpn_backbone(feats)
        feats = view_selector(feats)

    feature_maps = list(zip(backbone.out_strides, feats))
    plot_features(
        [(name, images[name]) for name in selected_cams],
        feature_maps,
        channels=args.channels,
        cols=args.cols,
        output=output_path,
    )

    projection = fuse_projection(inputs, cameras=selected_cams)
    projection_batch = ProjectionBatch.from_samples([projection]).to(device)

    # 只有在按 batch 维度拼接视角时 (cat_dim=0) 才能进行 BEV 融合
    if len(selected_cams) > 1 and args.cat_dim != 0:
        print("[warn] 多视角 BEV 融合需要 --cat-dim 0，已跳过 DeformablePointSample")
        return

    sampler = DeformablePointSample(
        point_cloud_range=args.point_cloud_range,
        voxel_size=args.voxel_size,
        num_cams=len(selected_cams),
        embed_dims=embed_dims,
        num_levels=len(feats),
    ).to(device)
    sampler.eval()

    imgs_tensor = torch.stack([images[name] for name in selected_cams], dim=0).to(device)
    mlvl_feats = [feat for feat in feats]
    bev_feats, world_pts, cam_pts = sampler(
        mlvl_feats,
        projection_batch,
        intrinsics=projection_batch.intrinsics,
        imgs=imgs_tensor,
    )
    print(f"[info] BEV feature volume shape: {tuple(bev_feats.shape)}")

    anchor_generator = AlignedAnchor3DRangeGenerator()
    anchors = anchor_generator.anchors_single_range(
        sampler.voxel_shape,
        args.point_cloud_range,
        device=device,
    )
    print(f"[info] Generated anchors shape: {tuple(anchors.shape)}")

    if args.bev_output:
        if bev_feats.dim() != 5:
            raise ValueError("BEV visualization expects features with shape (N, C, D, H, W)")
        if args.bev_reduction == "mean":
            bev_map = bev_feats.mean(dim=2)
        else:
            bev_map = bev_feats.max(dim=2).values
        channel = max(0, min(args.bev_channel, bev_map.shape[1] - 1))
        bev_slice = bev_map[0, channel].detach().cpu()
        bev_norm = normalize_feature(bev_slice)
        bev_img = bev_norm.numpy()
        bev_output = Path(args.bev_output).expanduser()
        bev_output.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 5))
        plt.imshow(bev_img, cmap=args.bev_colormap)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(bev_output, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[info] Saved BEV heatmap to {bev_output}")


if __name__ == "__main__":
    main()
