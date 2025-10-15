import torch

from pkt.data.nuscenes import NuScenesLidarFusionDataset


class FakeNuScenes:
    def __init__(self, dataroot: str) -> None:
        self.dataroot = dataroot
        self.scene = [
            {
                "token": "scene-1",
                "name": "scene-1",
                "first_sample_token": "sample-1",
            }
        ]
        self.sample = [
            {
                "token": "sample-1",
                "scene_token": "scene-1",
                "timestamp": 123456,
                "data": {
                    "LIDAR_TOP": "lidar-sd-1",
                    "CAM_FRONT": "cam-sd-1",
                },
                "next": "",
                "anns": ["ann-1"],
            }
        ]
        self.sample_data = {
            "lidar-sd-1": {
                "token": "lidar-sd-1",
                "ego_pose_token": "ego-1",
                "calibrated_sensor_token": "calib-lidar",
                "filename": "samples/LIDAR_TOP/lidar.pcd.bin",
            },
            "cam-sd-1": {
                "token": "cam-sd-1",
                "ego_pose_token": "ego-1",
                "calibrated_sensor_token": "calib-cam",
                "filename": "samples/CAM_FRONT/image.jpg",
            },
        }
        self.ego_pose = {
            "ego-1": {
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 0.0],
            }
        }
        self.calibrated_sensor = {
            "calib-lidar": {
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 0.0],
                "camera_intrinsic": None,
            },
            "calib-cam": {
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 0.0],
                "camera_intrinsic": [
                    [1000.0, 0.0, 640.0],
                    [0.0, 1000.0, 360.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        }
        self.sample_annotation = {
            "ann-1": {
                "token": "ann-1",
                "category_name": "vehicle.car",
                "translation": [1.0, 2.0, 3.0],
                "size": [4.0, 1.5, 1.4],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "attribute_tokens": ["attr-1"],
                "instance_token": "instance-1",
            }
        }

    def get(self, table_name: str, token: str):
        table = getattr(self, table_name)
        if isinstance(table, list):
            for item in table:
                if item["token"] == token:
                    return item
            raise KeyError(f"{token} not in table {table_name}")
        return table[token]


def test_nuscenes_dataset_basic(tmp_path):
    nusc = FakeNuScenes(str(tmp_path))

    def lidar_loader(sample_rec, lidar_sd_rec, nusc_instance):
        return torch.ones((5, 4), dtype=torch.float32)

    def image_loader(sample_rec, cam_sd_rec, nusc_instance):
        return torch.full((3, 2, 2), 0.5, dtype=torch.float32)

    dataset = NuScenesLidarFusionDataset(
        dataroot=str(tmp_path),
        version="v1.0-mini",
        split="train",
        nusc=nusc,
        scene_names=["scene-1"],
        camera_channels=["CAM_FRONT"],
        lidar_loader=lidar_loader,
        image_loader=image_loader,
        load_annotations=True,
        skip_empty=True,
        random_seed=42,
    )

    assert len(dataset) == 1

    inputs, target = dataset[0]

    assert inputs["lidar"].shape == (5, 4)
    assert "CAM_FRONT" in inputs["images"]
    assert torch.allclose(inputs["images"]["CAM_FRONT"], torch.full((3, 2, 2), 0.5))

    metadata = inputs["metadata"]
    assert metadata["sample_token"] == "sample-1"
    assert metadata["lidar"]["calibration"].rotation.shape == (4,)
    camera_meta = metadata["cameras"]["CAM_FRONT"]
    assert camera_meta["calibration"].camera_intrinsic.shape == (3, 3)

    assert target is not None
    assert target["labels"].shape == (1,)
    assert target["boxes_3d"].shape == (1, 10)
    assert target["annotation_tokens"] == ["ann-1"]
    assert target["instance_tokens"] == ["instance-1"]
    assert target["category_map"]["car"] == 0
    assert target["sample_token"] == "sample-1"
    assert target["scene_token"] == "scene-1"
