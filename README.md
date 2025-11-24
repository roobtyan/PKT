# PKT

感知工具库，现包含一个基于配置文件驱动的深度学习训练框架示例实现。

## 快速开始

1. 安装依赖（需要 Python 3.10+，并预先安装 [PyTorch](https://pytorch.org/)）。
   ```bash
   pip install pyyaml torch
   ```
2. 运行示例训练：
   ```bash
   python scripts/train.py configs/random_classification.yaml
   ```
   该配置会使用随机生成的数据训练一个简单的多层感知机并输出训练日志。

## 目录结构

- `pkt/`：框架源码，包含配置解析、注册表、模型、优化器和训练引擎等模块。
- `configs/`：示例配置文件。
- `scripts/`：命令行工具。
- `tests/`：基础单元测试，验证配置加载与模型构建流程。

更多设计细节可参考 `docs/config_framework_plan.md`。

## NuScenes 数据加载器

- 新增 `nuscenes_lidar_fusion` 数据集，可从 NuScenes 读取激光雷达 + 多摄像头数据并返回用于融合模型的输入与 3D 标注。
- 依赖 `nuscenes-devkit` 以及用于加载图片的 `Pillow`，请提前安装：`pip install nuscenes-devkit pillow`。
- 配置示例（仅示意，请按实际路径调整）：

  ```yaml
  datasets:
    train:
      name: nuscenes_lidar_fusion
      params:
        dataroot: /data/nuscenes
        version: v1.0-mini
        split: train
        camera_channels: [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT]
        num_sweeps: 1
        max_lidar_points: 80000
        skip_empty: true
        random_seed: 42
    val:
      name: nuscenes_lidar_fusion
      params:
        dataroot: /data/nuscenes
        version: v1.0-mini
        split: val
        camera_channels: [CAM_FRONT]
        load_annotations: false  # 推理或无标注场景时可关闭
  ```

- 在代码中构建数据集时可以传入 `transform`、`target_transform` 以及自定义的 `lidar_loader`/`image_loader` 回调，以适配具体的融合模型前处理流程。
- 数据准备：将官方的 `v1.0-trainval_meta.tgz` 解压到 `dataroot`，再按顺序解压 `v1.0-trainval0X_blobs.tgz`（本仓库提供 `scripts/visualize_nuscenes.py` 可辅助完成解压和可视化）。例如：
  ```bash
  # 解压单个分卷并生成一张样例可视化图（包含 LiDAR BEV + 某个摄像头视角）
  python scripts/visualize_nuscenes.py \
    --dataroot /data/nuscenes \
    --version v1.0-trainval \
    --split train \
    --blob /path/to/v1.0-trainval03_blobs.tgz \
    --index 0 \
    --output outputs/nuscenes_sample.png
  ```
  如果数据已经解压，可去掉 `--blob` 直接运行。
