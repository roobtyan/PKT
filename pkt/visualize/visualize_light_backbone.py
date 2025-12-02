#!/usr/bin/env python
"""Visualize LightBackbone feature maps on a NuScenes sample."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pkt.data.nuscenes import NuScenesLidarFusionDataset
from pkt.models.light_backbone import LightBackbone
from pkt.models.fpn import FPN


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
    backbone.to(device).eval()

    with torch.no_grad():
        if len(image_list) == 1:
            model_input = reference_image.unsqueeze(0).to(device)  # (B, C, H, W)
        else:
            model_input = [img.unsqueeze(0).to(device) for img in image_list]  # sequence input
        feats = backbone(model_input)
        feats = fpn_backbone(feats)

    feature_maps = list(zip(backbone.out_strides, feats))
    plot_features(
        [(name, images[name]) for name in selected_cams],
        feature_maps,
        channels=args.channels,
        cols=args.cols,
        output=output_path,
    )


if __name__ == "__main__":
    main()
