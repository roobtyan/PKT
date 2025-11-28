#!/usr/bin/env python
"""Extract a nuScenes blob archive and visualize one sample.

Usage example:
    python scripts/visualize_nuscenes.py \
        --dataroot /data/nuscenes \
        --version v1.0-trainval \
        --split train \
        --blob /data/v1.0-trainval03_blobs.tgz \
        --index 0 \
        --output outputs/nuscenes_sample.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tarfile
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pkt import DEFAULT_CAMERA_CHANNELS, NuScenesLidarFusionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unpack v1.0-trainvalXX blobs and render a nuScenes sample."
    )
    parser.add_argument("--dataroot", type=str, required=True, help="nuScenes root directory.")
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        help="nuScenes version string (e.g. v1.0-trainval or v1.0-mini).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split key understood by nuscenes-devkit (train/val/mini_train/mini_val/etc.).",
    )
    parser.add_argument(
        "--blob",
        type=str,
        default=None,
        help="Optional path to a v1.0-trainvalXX_blobs.tgz archive to extract into dataroot.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Extract even if target folders already exist in dataroot.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Dataset index to visualize.",
    )
    parser.add_argument(
        "--num-sweeps",
        type=int,
        default=1,
        help="How many LiDAR sweeps to aggregate per sample.",
    )
    parser.add_argument(
        "--max-lidar-points",
        type=int,
        default=150_000,
        help="Randomly subsample LiDAR points to this count to keep plotting light.",
    )
    parser.add_argument(
        "--camera-channels",
        type=str,
        nargs="+",
        default=list(DEFAULT_CAMERA_CHANNELS),
        help="Camera channels to load alongside LiDAR.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/nuscenes_sample.png",
        help="Where to save the composite visualization.",
    )
    return parser.parse_args()


def _is_within_directory(base: Path, target: Path) -> bool:
    base = base.resolve()
    try:
        target = target.resolve()
    except FileNotFoundError:
        # Resolve raises for non-existent path; still perform a best-effort join check.
        target = (base / target).resolve()
    return os.path.commonpath([base]) == os.path.commonpath([base, target])


def extract_blob_archive(blob_path: Path, dataroot: Path, *, force: bool = False) -> None:
    """Extract a nuScenes blob archive into the dataroot."""

    if not blob_path.exists():
        raise FileNotFoundError(f"Blob archive not found: {blob_path}")

    dataroot.mkdir(parents=True, exist_ok=True)
    with tarfile.open(blob_path, "r:gz") as tar:
        top_level_entries = {Path(member.name).parts[0] for member in tar.getmembers() if member.name}
        if (
                not force
                and top_level_entries
                and all((dataroot / entry).exists() for entry in top_level_entries)
        ):
            print(f"[info] Skip extraction: detected existing entries under {dataroot}")
            return

        for member in tar.getmembers():
            member_path = dataroot / member.name
            if not _is_within_directory(dataroot, member_path):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        print(f"[info] Extracting {blob_path.name} into {dataroot} ...")
        tar.extractall(path=dataroot)
        print("[info] Extraction finished.")


def quaternion_yaw(q: Iterable[float]) -> float:
    """Return heading (around z) from a quaternion [w, x, y, z]."""

    w, x, y, z = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def bev_box_corners(box: np.ndarray) -> np.ndarray:
    """Compute BEV corners (4, 2) from a nuScenes box array [x, y, z, w, l, h, qw, qx, qy, qz]."""

    x, y, _, width, length, _, qw, qx, qy, qz = box
    yaw = quaternion_yaw((qw, qx, qy, qz))
    dx = length * 0.5
    dy = width * 0.5
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    rotated = corners @ rotation.T
    rotated[:, 0] += x
    rotated[:, 1] += y
    return rotated


def plot_sample(
        lidar: torch.Tensor,
        images: dict[str, torch.Tensor],
        target: dict | None,
        *,
        out_path: Path,
) -> None:
    """Create a composite figure with BEV LiDAR and one camera view."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lidar_np = lidar.detach().cpu().numpy()
    xy = lidar_np[:, :2]
    z = lidar_np[:, 2] if lidar_np.shape[1] > 2 else np.zeros(len(lidar_np))

    boxes = target.get("boxes_3d").cpu().numpy() if target and "boxes_3d" in target else None
    labels = target.get("labels").cpu().numpy() if target and "labels" in target else None

    # Choose the first camera alphabetically for the side view.
    cam_name = sorted(images.keys())[0]
    cam_tensor = images[cam_name].detach().cpu()
    cam_np = cam_tensor.permute(1, 2, 0).numpy().clip(0.0, 1.0)

    fig, (ax_bev, ax_cam) = plt.subplots(1, 2, figsize=(14, 7))

    ax_bev.scatter(xy[:, 0], xy[:, 1], c=z, s=0.2, cmap="viridis", alpha=0.9)
    if boxes is not None:
        for idx, box in enumerate(boxes):
            corners = bev_box_corners(box)
            xs, ys = corners[:, 0], corners[:, 1]
            ax_bev.plot(
                np.r_[xs, xs[0]],
                np.r_[ys, ys[0]],
                color="tomato",
                linewidth=1.2,
            )
            if labels is not None:
                label = int(labels[idx])
                ax_bev.text(
                    box[0],
                    box[1],
                    str(label),
                    color="white",
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="tomato", alpha=0.6, edgecolor="none"),
                )
    ax_bev.set_aspect("equal")
    ax_bev.set_title("LiDAR BEV (x forward, y left)")
    ax_bev.set_xlabel("x [m]")
    ax_bev.set_ylabel("y [m]")
    ax_bev.grid(True, linestyle="--", linewidth=0.4)

    ax_cam.imshow(cam_np)
    ax_cam.axis("off")
    ax_cam.set_title(cam_name)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] Saved visualization to {out_path}")


def main() -> None:
    args = parse_args()
    dataroot = Path(args.dataroot).expanduser().resolve()

    if args.blob:
        extract_blob_archive(Path(args.blob).expanduser().resolve(), dataroot, force=args.force_extract)

    dataset = NuScenesLidarFusionDataset(
        dataroot=str(dataroot),
        version=args.version,
        split=args.split,
        camera_channels=args.camera_channels,
        num_sweeps=args.num_sweeps,
        max_lidar_points=args.max_lidar_points,
        load_annotations=True,
        skip_empty=False,
    )

    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index {args.index} out of range (dataset has {len(dataset)} samples)")

    inputs, target = dataset[args.index]
    plot_sample(inputs["lidar"], inputs["images"], target, out_path=Path(args.output))


if __name__ == "__main__":
    main()
