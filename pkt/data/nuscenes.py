"""NuScenes dataset utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from pkt.data.datasets import BaseDataset, DATASET_REGISTRY


DEFAULT_CAMERA_CHANNELS: Tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

DEFAULT_CLASS_NAMES: Tuple[str, ...] = (
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
)


def _to_tensor(data: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert nested data to a torch tensor."""

    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    array = np.asarray(data, dtype=np.float32 if dtype == torch.float32 else None)
    if array.ndim == 0:
        array = np.expand_dims(array, 0)
    return torch.from_numpy(array).to(dtype=dtype)


def _normalize_category(name: str) -> str:
    """Normalize NuScenes category strings to their leaf name."""

    return name.split(".")[-1]


def _list_to_tensor(items: List[List[float]], *, features: int, dtype: torch.dtype) -> torch.Tensor:
    if not items:
        return torch.zeros((0, features), dtype=dtype)
    return torch.tensor(items, dtype=dtype)


@dataclass
class SensorCalibration:
    rotation: torch.Tensor
    translation: torch.Tensor
    camera_intrinsic: torch.Tensor | None

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SensorCalibration":
        rotation = _to_tensor(record["rotation"], dtype=torch.float32)
        translation = _to_tensor(record["translation"], dtype=torch.float32)
        camera_intrinsic = None
        if record.get("camera_intrinsic") is not None:
            camera_intrinsic = _to_tensor(record["camera_intrinsic"], dtype=torch.float32)
        return cls(rotation=rotation, translation=translation, camera_intrinsic=camera_intrinsic)


@dataclass
class EgoPose:
    rotation: torch.Tensor
    translation: torch.Tensor

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "EgoPose":
        rotation = _to_tensor(record["rotation"], dtype=torch.float32)
        translation = _to_tensor(record["translation"], dtype=torch.float32)
        return cls(rotation=rotation, translation=translation)


class NuScenesUnavailableError(ImportError):
    """Raised when ``nuscenes-devkit`` is missing."""


def _load_nuscenes(version: str, dataroot: Path) -> Any:
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:  # pragma: no cover - import failure only hit in prod if missing dependency
        raise NuScenesUnavailableError(
            "NuScenes dataset requires the 'nuscenes-devkit' package. "
            "Install it with `pip install nuscenes-devkit` or provide a pre-initialized NuScenes "
            "instance via the `nusc` parameter."
        ) from exc
    return NuScenes(version=version, dataroot=str(dataroot), verbose=False)


@DATASET_REGISTRY.register("nuscenes_lidar_fusion")
class NuScenesLidarFusionDataset(BaseDataset):
    """Dataset that prepares LiDAR + camera inputs from NuScenes samples.

    Parameters
    ----------
    dataroot:
        Root directory of the NuScenes dataset.
    version:
        Dataset version string (e.g. ``"v1.0-mini"`` or ``"v1.0-trainval"``).
    split:
        Split identifier. Must correspond to a key in :func:`nuscenes.utils.splits.create_splits_scenes`
        (e.g. ``"train"``, ``"val"``, ``"mini_train"``). When using the mini set, plain split names are
        automatically mapped (``"train"`` -> ``"mini_train"`` etc.).
    lidar_channel:
        Primary LiDAR sensor channel to load. Defaults to ``"LIDAR_TOP"``.
    camera_channels:
        Iterable of camera channels to load. Defaults to the six surround cameras.
    num_sweeps:
        Number of LiDAR sweeps to aggregate. Values greater than 1 require ``nuscenes-devkit``.
    max_lidar_points:
        Optional upper bound on returned LiDAR points. When provided, points are randomly subsampled.
    class_names:
        Iterable of class names to keep in the annotations. Defaults to the NuScenes detection classes.
    load_annotations:
        If ``False`` the dataset returns ``target=None`` to support unlabeled evaluation.
    skip_empty:
        Skip samples that do not contain any annotation matching ``class_names``.
    transform:
        Optional callable applied as ``transform(inputs, target)`` before returning.
    target_transform:
        Optional callable applied to the target dictionary.
    scene_names:
        Explicit list of scene names to use. Overrides ``split``.
    sample_tokens:
        Explicit list of sample tokens to use. Overrides ``split`` and ``scene_names``.
    nusc:
        Optionally provide a pre-initialised :class:`nuscenes.nuscenes.NuScenes` instance. Useful for tests.
    random_seed:
        Seed for LiDAR subsampling.
    lidar_loader:
        Optional callable ``lidar_loader(sample_rec, lidar_sd_rec, nusc) -> torch.Tensor`` to override LiDAR loading.
    image_loader:
        Optional callable ``image_loader(sample_rec, image_sd_rec, nusc) -> torch.Tensor`` to override image loading.
    """

    def __init__(
        self,
        dataroot: str,
        version: str,
        split: str,
        *,
        lidar_channel: str = "LIDAR_TOP",
        camera_channels: Sequence[str] | None = None,
        num_sweeps: int = 1,
        max_lidar_points: int | None = None,
        class_names: Sequence[str] | None = None,
        load_annotations: bool = True,
        skip_empty: bool = False,
        transform: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Tuple[Any, Any]] | None = None,
        target_transform: Callable[[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] | None = None,
        scene_names: Sequence[str] | None = None,
        sample_tokens: Sequence[str] | None = None,
        nusc: Any | None = None,
        random_seed: int | None = None,
        lidar_loader: Callable[[Mapping[str, Any], Mapping[str, Any], Any], torch.Tensor] | None = None,
        image_loader: Callable[[Mapping[str, Any], Mapping[str, Any], Any], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.dataroot = Path(dataroot)
        if not self.dataroot.exists():
            raise FileNotFoundError(f"NuScenes dataroot not found: {self.dataroot}")
        self.version = version
        self.split = split
        self.lidar_channel = lidar_channel
        self.camera_channels = tuple(camera_channels or DEFAULT_CAMERA_CHANNELS)
        if not self.camera_channels:
            raise ValueError("camera_channels must contain at least one entry")
        self.num_sweeps = max(1, int(num_sweeps))
        self.max_lidar_points = int(max_lidar_points) if max_lidar_points is not None else None
        self.load_annotations = bool(load_annotations)
        self.skip_empty = bool(skip_empty)
        self.transform = transform
        self.target_transform = target_transform
        self.class_names = tuple(class_names or DEFAULT_CLASS_NAMES)
        if not self.class_names:
            raise ValueError("class_names must contain at least one entry")
        self.class_map = {name: idx for idx, name in enumerate(self.class_names)}
        self.lidar_loader = lidar_loader
        self.image_loader = image_loader
        self.generator = torch.Generator()
        if random_seed is not None:
            self.generator.manual_seed(random_seed)

        self.nusc = nusc if nusc is not None else _load_nuscenes(version=version, dataroot=self.dataroot)

        self.sample_tokens: List[str] = self._resolve_sample_tokens(
            split=split, scene_names=scene_names, provided_tokens=sample_tokens
        )
        if not self.sample_tokens:
            raise ValueError(
                "No NuScenes samples matched the provided split/filters. "
                "Check 'split', 'scene_names', or 'sample_tokens'."
            )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.sample_tokens)

    def __getitem__(self, index: int) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:  # type: ignore[override]
        sample_token = self.sample_tokens[index]
        sample_rec = self.nusc.get("sample", sample_token)

        lidar_tensor, lidar_meta = self._load_lidar(sample_rec)
        images, calibrations, ego_poses = self._load_images(sample_rec)

        metadata = {
            "sample_token": sample_token,
            "scene_token": sample_rec["scene_token"],
            "timestamp": sample_rec["timestamp"],
            "lidar": {**lidar_meta},
            "cameras": {
                cam: {
                    "ego_pose": ego_poses[cam],
                    "calibration": calibrations[cam],
                }
                for cam in self.camera_channels
            },
        }

        inputs: Dict[str, Any] = {
            "lidar": lidar_tensor,
            "images": images,
            "metadata": metadata,
        }

        target: Optional[Dict[str, Any]] = None
        if self.load_annotations:
            target = self._load_annotations(sample_rec)
            if self.target_transform is not None:
                target = self.target_transform(target)

        if self.transform is not None:
            inputs, target = self.transform(inputs, target)

        return inputs, target

    # --------------------------------------------------------------------- #
    # Initialisation helpers

    def _resolve_sample_tokens(
        self,
        *,
        split: str,
        scene_names: Sequence[str] | None,
        provided_tokens: Sequence[str] | None,
    ) -> List[str]:
        if provided_tokens is not None:
            return list(provided_tokens)
        if scene_names is None:
            scene_names = self._scene_names_from_split(split)
        tokens = self._tokens_from_scene_names(scene_names)
        if self.skip_empty and self.load_annotations:
            tokens = [token for token in tokens if self._has_valid_annotations(token)]
        return tokens

    def _scene_names_from_split(self, split: str) -> Sequence[str]:
        split_key_candidates = self._possible_split_keys(split)
        try:
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError as exc:  # pragma: no cover - guarded by tests providing scene_names
            raise NuScenesUnavailableError(
                "Resolving a split requires 'nuscenes-devkit'. Install it or provide `scene_names`."
            ) from exc

        split_map = create_splits_scenes()
        for key in split_key_candidates:
            if key in split_map:
                return split_map[key]
        available = ", ".join(sorted(split_map))
        raise KeyError(f"Unknown NuScenes split '{split}'. Available splits: {available}")

    def _possible_split_keys(self, split: str) -> List[str]:
        split = split.strip()
        keys = [split]
        if not split.startswith("mini_"):
            keys.append(f"mini_{split}")
        if not split.endswith("_detect"):
            keys.append(f"{split}_detect")
        return keys

    def _tokens_from_scene_names(self, scene_names: Sequence[str]) -> List[str]:
        scene_name_set = set(scene_names)
        if not scene_name_set:
            raise ValueError("scene_names must contain at least one entry")
        tokens: List[str] = []
        for scene_rec in self.nusc.scene:
            if scene_rec["name"] not in scene_name_set:
                continue
            current = scene_rec["first_sample_token"]
            while current:
                tokens.append(current)
                sample_rec = self.nusc.get("sample", current)
                current = sample_rec["next"]
        return tokens

    def _has_valid_annotations(self, sample_token: str) -> bool:
        sample_rec = self.nusc.get("sample", sample_token)
        for ann_token in sample_rec.get("anns", []):
            ann_rec = self.nusc.get("sample_annotation", ann_token)
            if _normalize_category(ann_rec["category_name"]) in self.class_map:
                return True
        return False

    # --------------------------------------------------------------------- #
    # Data loading helpers

    def _load_lidar(self, sample_rec: Mapping[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        lidar_token = sample_rec["data"].get(self.lidar_channel)
        if lidar_token is None:
            raise KeyError(f"Sample {sample_rec['token']} does not contain channel '{self.lidar_channel}'")
        lidar_sd_rec = self.nusc.get("sample_data", lidar_token)

        if self.lidar_loader is not None:
            lidar_tensor = self.lidar_loader(sample_rec, lidar_sd_rec, self.nusc)
        else:
            lidar_tensor = self._default_lidar_loader(sample_rec, lidar_sd_rec)

        if not isinstance(lidar_tensor, torch.Tensor):
            lidar_tensor = _to_tensor(lidar_tensor, dtype=torch.float32)
        if lidar_tensor.ndim != 2:
            raise ValueError(f"Expected LiDAR tensor with shape (N, C); got shape {tuple(lidar_tensor.shape)}")

        num_points = lidar_tensor.shape[0]
        if self.max_lidar_points is not None and num_points > self.max_lidar_points:
            indices = torch.randperm(num_points, generator=self.generator)[: self.max_lidar_points]
            lidar_tensor = lidar_tensor.index_select(0, indices)

        ego_pose = EgoPose.from_record(self.nusc.get("ego_pose", lidar_sd_rec["ego_pose_token"]))
        calibration = SensorCalibration.from_record(
            self.nusc.get("calibrated_sensor", lidar_sd_rec["calibrated_sensor_token"])
        )

        metadata = {
            "ego_pose": ego_pose,
            "calibration": calibration,
            "filename": str((self.dataroot / lidar_sd_rec["filename"]).resolve()),
        }
        return lidar_tensor, metadata

    def _default_lidar_loader(
        self,
        sample_rec: Mapping[str, Any],
        lidar_sd_rec: Mapping[str, Any],
    ) -> torch.Tensor:
        try:
            from nuscenes.utils.data_classes import LidarPointCloud
        except ImportError as exc:  # pragma: no cover - triggered when nuscenes-devkit missing at runtime
            raise NuScenesUnavailableError(
                "Loading LiDAR data requires the 'nuscenes-devkit' package or a custom `lidar_loader`."
            ) from exc

        lidar_path = self.dataroot / lidar_sd_rec["filename"]
        if self.num_sweeps > 1:
            pc, times = LidarPointCloud.from_file_multisweep(
                self.nusc,
                sample_rec,
                chan=self.lidar_channel,
                ref_chan=self.lidar_channel,
                nsweeps=self.num_sweeps,
            )
            points = pc.points.T.astype(np.float32)
            times = np.asarray(times, dtype=np.float32).reshape(-1, 1)
            lidar_array = np.concatenate([points, times], axis=1)
        else:
            pc = LidarPointCloud.from_file(str(lidar_path))
            lidar_array = pc.points.T.astype(np.float32)
        return torch.from_numpy(lidar_array)

    def _load_images(
        self,
        sample_rec: Mapping[str, Any],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, SensorCalibration], Dict[str, EgoPose]]:
        images: Dict[str, torch.Tensor] = {}
        calibrations: Dict[str, SensorCalibration] = {}
        ego_poses: Dict[str, EgoPose] = {}
        for cam in self.camera_channels:
            cam_token = sample_rec["data"].get(cam)
            if cam_token is None:
                raise KeyError(f"Sample {sample_rec['token']} missing camera channel '{cam}'")
            cam_sd_rec = self.nusc.get("sample_data", cam_token)

            if self.image_loader is not None:
                image_tensor = self.image_loader(sample_rec, cam_sd_rec, self.nusc)
            else:
                image_tensor = self._default_image_loader(cam_sd_rec)

            if not isinstance(image_tensor, torch.Tensor):
                image_tensor = _to_tensor(image_tensor, dtype=torch.float32)

            if image_tensor.ndim != 3:
                raise ValueError(f"Expected camera tensor with shape (C, H, W); got shape {tuple(image_tensor.shape)}")

            if image_tensor.dtype != torch.float32:
                image_tensor = image_tensor.to(torch.float32)

            images[cam] = image_tensor
            calibrations[cam] = SensorCalibration.from_record(
                self.nusc.get("calibrated_sensor", cam_sd_rec["calibrated_sensor_token"])
            )
            ego_poses[cam] = EgoPose.from_record(self.nusc.get("ego_pose", cam_sd_rec["ego_pose_token"]))

        return images, calibrations, ego_poses

    def _default_image_loader(self, cam_sd_rec: Mapping[str, Any]) -> torch.Tensor:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - triggered when Pillow missing
            raise ImportError(
                "Loading camera images requires the 'Pillow' package or a custom `image_loader`."
            ) from exc
        image_path = self.dataroot / cam_sd_rec["filename"]
        with Image.open(image_path) as img:
            image_array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        return image_tensor

    def _load_annotations(self, sample_rec: Mapping[str, Any]) -> Dict[str, Any]:
        ann_tokens = sample_rec.get("anns", [])
        boxes: List[List[float]] = []
        labels: List[int] = []
        attribute_tokens: List[str] = []
        instance_tokens: List[str] = []

        valid_ann_tokens: List[str] = []
        for ann_token in ann_tokens:
            ann_rec = self.nusc.get("sample_annotation", ann_token)
            category = _normalize_category(ann_rec["category_name"])
            if category not in self.class_map:
                continue
            translation = ann_rec["translation"]
            size = ann_rec["size"]
            rotation = ann_rec["rotation"]
            boxes.append([*translation, *size, *rotation])
            labels.append(self.class_map[category])
            attribute_tokens.append(ann_rec.get("attribute_tokens", []))
            instance_tokens.append(ann_rec["instance_token"])
            valid_ann_tokens.append(ann_token)

        target: Dict[str, Any] = {
            "boxes_3d": _list_to_tensor(boxes, features=10, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "annotation_tokens": valid_ann_tokens,
            "instance_tokens": instance_tokens,
            "category_map": self.class_map,
            "attribute_tokens": attribute_tokens,
            "sample_token": sample_rec["token"],
            "scene_token": sample_rec["scene_token"],
        }

        if target["labels"].numel() == 0:
            target["labels"] = torch.zeros((0,), dtype=torch.long)

        return target


def _quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    if quat.numel() != 4:
        raise ValueError(f"Quaternion must have 4 values; got shape {tuple(quat.shape)}")
    w, x, y, z = quat.tolist()
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return torch.tensor(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=torch.float32,
    )


def _build_transform(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    mat = torch.eye(4, dtype=torch.float32)
    mat[:3, :3] = _quaternion_to_matrix(rotation)
    mat[:3, 3] = translation
    return mat


def _invert_transform(mat: torch.Tensor) -> torch.Tensor:
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    inv = torch.eye(4, dtype=torch.float32)
    inv[:3, :3] = rot.t()
    inv[:3, 3] = -rot.t() @ trans
    return inv


@dataclass
class ProjectionSample:
    cameras: List[str]
    proj_mats: torch.Tensor  # (num_cams, 4, 4)
    extrinsics: torch.Tensor  # (num_cams, 4, 4)
    intrinsics: torch.Tensor  # (num_cams, 3, 3)
    image_sizes: torch.Tensor  # (num_cams, 2) -> (H, W)

    def to(self, device: torch.device | str) -> "ProjectionSample":
        return ProjectionSample(
            cameras=list(self.cameras),
            proj_mats=self.proj_mats.to(device),
            extrinsics=self.extrinsics.to(device),
            intrinsics=self.intrinsics.to(device),
            image_sizes=self.image_sizes.to(device),
        )


@dataclass
class ProjectionBatch:
    cameras: List[str]
    proj_mats: torch.Tensor  # (B, num_cams, 4, 4)
    extrinsics: torch.Tensor  # (B, num_cams, 4, 4)
    intrinsics: torch.Tensor  # (B, num_cams, 3, 3)
    image_sizes: torch.Tensor  # (B, num_cams, 2)

    def to(self, device: torch.device | str) -> "ProjectionBatch":
        return ProjectionBatch(
            cameras=list(self.cameras),
            proj_mats=self.proj_mats.to(device),
            extrinsics=self.extrinsics.to(device),
            intrinsics=self.intrinsics.to(device),
            image_sizes=self.image_sizes.to(device),
        )

    @classmethod
    def from_samples(cls, samples: Sequence[ProjectionSample]) -> "ProjectionBatch":
        if not samples:
            raise ValueError("ProjectionBatch.from_samples requires at least one sample")
        cameras = samples[0].cameras
        for sample in samples:
            if sample.cameras != cameras:
                raise ValueError("All samples must share the same camera ordering for batching")
        proj = torch.stack([sample.proj_mats for sample in samples], dim=0)
        extr = torch.stack([sample.extrinsics for sample in samples], dim=0)
        intr = torch.stack([sample.intrinsics for sample in samples], dim=0)
        sizes = torch.stack([sample.image_sizes for sample in samples], dim=0)
        return cls(cameras=list(cameras), proj_mats=proj, extrinsics=extr, intrinsics=intr, image_sizes=sizes)


def _lidar_to_image_matrix(
    lidar_global: torch.Tensor,
    cam_pose: torch.Tensor,
    cam_calib: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_to_cam_ego = _invert_transform(cam_pose)
    ego_to_cam = _invert_transform(cam_calib)
    lidar_to_cam = ego_to_cam @ global_to_cam_ego @ lidar_global

    proj = torch.zeros((4, 4), dtype=torch.float32)
    proj[:3, :4] = intrinsics @ lidar_to_cam[:3, :4]
    proj[3, 3] = 1.0
    return lidar_to_cam, proj


def fuse_projection(
    inputs: Mapping[str, Any],
    cameras: Optional[Sequence[str]] = None,
) -> ProjectionSample:
    if "metadata" not in inputs or "images" not in inputs:
        raise KeyError("Inputs must contain 'metadata' and 'images' to build projections")

    metadata: Mapping[str, Any] = inputs["metadata"]
    images: Mapping[str, torch.Tensor] = inputs["images"]

    lidar_meta = metadata["lidar"]  # type: ignore[index]
    lidar_calibration: SensorCalibration = lidar_meta["calibration"]  # type: ignore[assignment]
    lidar_pose: EgoPose = lidar_meta["ego_pose"]  # type: ignore[assignment]

    lidar_to_ego = _build_transform(_to_tensor(lidar_calibration.rotation), _to_tensor(lidar_calibration.translation))
    ego_to_global = _build_transform(_to_tensor(lidar_pose.rotation), _to_tensor(lidar_pose.translation))
    lidar_to_global = ego_to_global @ lidar_to_ego

    camera_meta: Mapping[str, Mapping[str, Any]] = metadata["cameras"]  # type: ignore[index]
    if cameras is None:
        camera_list = sorted(camera_meta.keys())
    else:
        camera_list = list(cameras)
    if not camera_list:
        raise ValueError("No cameras provided for projection fusion")

    proj_mats: List[torch.Tensor] = []
    extrinsics: List[torch.Tensor] = []
    intrinsics: List[torch.Tensor] = []
    image_sizes: List[torch.Tensor] = []

    for cam_name in camera_list:
        if cam_name not in camera_meta:
            raise KeyError(f"Camera '{cam_name}' not found in metadata")
        cam_info = camera_meta[cam_name]
        cam_pose: EgoPose = cam_info["ego_pose"]  # type: ignore[assignment]
        cam_calib: SensorCalibration = cam_info["calibration"]  # type: ignore[assignment]
        cam_pose_mat = _build_transform(_to_tensor(cam_pose.rotation), _to_tensor(cam_pose.translation))
        cam_calib_mat = _build_transform(_to_tensor(cam_calib.rotation), _to_tensor(cam_calib.translation))
        intrinsic = _to_tensor(cam_calib.camera_intrinsic, dtype=torch.float32)  # type: ignore[arg-type]
        if intrinsic.shape != (3, 3):
            raise ValueError(f"Invalid intrinsic matrix for '{cam_name}': shape {tuple(intrinsic.shape)}")

        lidar_to_cam, proj = _lidar_to_image_matrix(lidar_to_global, cam_pose_mat, cam_calib_mat, intrinsic)
        proj_mats.append(proj)
        extrinsics.append(lidar_to_cam)
        intrinsics.append(intrinsic)

        img_tensor = images[cam_name]
        image_sizes.append(torch.tensor(img_tensor.shape[-2:], dtype=torch.float32))

    return ProjectionSample(
        cameras=camera_list,
        proj_mats=torch.stack(proj_mats, dim=0),
        extrinsics=torch.stack(extrinsics, dim=0),
        intrinsics=torch.stack(intrinsics, dim=0),
        image_sizes=torch.stack(image_sizes, dim=0),
    )


def collate_and_fuse_projection(
    batch: Sequence[Mapping[str, Any]],
    cameras: Optional[Sequence[str]] = None,
) -> ProjectionBatch:
    samples = [fuse_projection(sample, cameras=cameras) for sample in batch]
    return ProjectionBatch.from_samples(samples)


class FuseProjection:
    """Dataset transform that appends fused projection matrices to inputs."""

    def __init__(self, cameras: Optional[Sequence[str]] = None) -> None:
        self.cameras = list(cameras) if cameras is not None else None

    def __call__(self, inputs: MutableMapping[str, Any], target: Any = None):
        projection = fuse_projection(inputs, cameras=self.cameras)
        new_inputs = dict(inputs)
        new_inputs["projection"] = projection
        return new_inputs, target


__all__ = [
    "NuScenesLidarFusionDataset",
    "DEFAULT_CLASS_NAMES",
    "DEFAULT_CAMERA_CHANNELS",
    "ProjectionSample",
    "ProjectionBatch",
    "FuseProjection",
    "collate_and_fuse_projection",
    "fuse_projection",
]
