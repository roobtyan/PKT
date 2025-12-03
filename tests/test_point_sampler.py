import torch

from pkt.data.nuscenes import ProjectionBatch, ProjectionSample
from pkt.models.anchors import AlignedAnchor3DRangeGenerator
from pkt.models.point_sample import DeformablePointSample


def test_point_sampler_forward():
    sampler = DeformablePointSample(
        point_cloud_range=(-2.0, -2.0, -1.0, 2.0, 2.0, 1.0),
        voxel_size=(2.0, 2.0, 1.0),
        num_cams=1,
        embed_dims=8,
        num_levels=1,
    )

    proj_sample = ProjectionSample(
        cameras=["cam"],
        proj_mats=torch.eye(4).unsqueeze(0),
        extrinsics=torch.eye(4).unsqueeze(0),
        intrinsics=torch.eye(3).unsqueeze(0),
        image_sizes=torch.tensor([[32.0, 32.0]]),
    )
    proj_batch = ProjectionBatch.from_samples([proj_sample])

    feature_map = torch.rand((1, 8, 16, 16))
    bev_feats, world_points, camera_points = sampler([feature_map], proj_batch, intrinsics=proj_batch.intrinsics)

    assert bev_feats.shape == (1, 8, 2, 2, 2)
    assert world_points.shape == (1, 2, 2, 2, 4)
    assert camera_points.shape == (1, 1, 2, 2, 2, 4)


def test_aligned_anchor_generator_shape():
    generator = AlignedAnchor3DRangeGenerator()
    anchors = generator.anchors_single_range(
        feature_size=(2, 2, 2),
        anchor_range=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
    )
    assert anchors.shape[-1] >= 7
    assert anchors.shape[:3] == (2, 2, 2)
