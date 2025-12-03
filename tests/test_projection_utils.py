import torch

from pkt.data.nuscenes import NuScenesLidarFusionDataset, ProjectionBatch, fuse_projection


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
                "anns": [],
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
                    [500.0, 0.0, 320.0],
                    [0.0, 500.0, 180.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        }

    def get(self, table_name: str, token: str):
        table = getattr(self, table_name)
        if isinstance(table, list):
            for item in table:
                if item["token"] == token:
                    return item
            raise KeyError(token)
        return table[token]


def test_fuse_projection_outputs(tmp_path):
    nusc = FakeNuScenes(str(tmp_path))

    def lidar_loader(sample_rec, lidar_sd_rec, nusc_instance):
        return torch.ones((4, 4), dtype=torch.float32)

    def image_loader(sample_rec, cam_sd_rec, nusc_instance):
        return torch.zeros((3, 2, 2), dtype=torch.float32)

    dataset = NuScenesLidarFusionDataset(
        dataroot=str(tmp_path),
        version="v1.0-mini",
        split="train",
        nusc=nusc,
        scene_names=["scene-1"],
        camera_channels=["CAM_FRONT"],
        lidar_loader=lidar_loader,
        image_loader=image_loader,
        load_annotations=False,
    )

    inputs, _ = dataset[0]
    projection = fuse_projection(inputs, cameras=["CAM_FRONT"])

    assert projection.proj_mats.shape == (1, 4, 4)
    assert projection.intrinsics.shape == (1, 3, 3)
    assert projection.image_sizes.tolist() == [[2.0, 2.0]]

    batch = ProjectionBatch.from_samples([projection, projection])
    assert batch.proj_mats.shape == (2, 1, 4, 4)
    assert batch.image_sizes.shape == (2, 1, 2)
