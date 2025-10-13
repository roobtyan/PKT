# 基于配置文件的深度学习框架方案

> 注：仓库中的 `pkt/` 目录已经基于本方案实现了一个最小化可运行版本，可结合代码与本文档一起阅读以了解整体设计与落地细节。

## 核心理念
- **解耦组件**：通过配置文件声明训练流程、数据管线、模型结构和输出头部，框架负责解析配置并实例化对应组件。
- **标准化接口**：约束各组件实现统一的构造函数与生命周期方法，保证可以通过反射/注册表机制动态加载。
- **可扩展的注册表**：提供模块化的 `Registry` 体系，允许用户将自定义的数据集、模型层、损失函数等注册后即可在配置中引用。

## 配置文件结构
建议使用 `YAML` 作为主配置格式，主要字段如下：

```yaml
experiment:
  name: exp_resnet50_cifar10
  work_dir: ./outputs/exp_resnet50_cifar10
  seed: 42

runtime:
  accelerator: gpu  # cpu/gpu/tpu/auto
  precision: fp16   # fp32/fp16/bf16
  grad_accum_steps: 1
  log_interval: 50
  val_interval: 1    # 单位：epoch

model:
  backbone:
    type: resnet50
    pretrained: true
    freeze_stages: 1
  head:
    type: classification
    num_classes: 10
    in_features: 2048
    loss:
      - type: cross_entropy
        weight: 1.0
      - type: label_smoothing
        eps: 0.1
  init:
    type: kaiming

optimizer:
  type: AdamW
  lr: 0.001
  weight_decay: 0.05
  betas: [0.9, 0.999]
  schedulers:
    - type: CosineAnnealingLR
      T_max: 200
    - type: WarmupLR
      warmup_epochs: 5

train_dataloader:
  dataset:
    type: CIFAR10
    root: ./data
    split: train
    download: true
    transforms:
      - type: RandomCrop
        size: 32
        padding: 4
      - type: RandomHorizontalFlip
      - type: ToTensor
      - type: Normalize
        mean: [0.4914, 0.4822, 0.4465]
        std: [0.2023, 0.1994, 0.2010]
  sampler:
    type: DistributedSampler
    shuffle: true
  dataloader:
    batch_size: 128
    num_workers: 8
    pin_memory: true
    drop_last: true

val_dataloader:
  dataset:
    type: CIFAR10
    root: ./data
    split: val
  sampler:
    type: SequentialSampler
  dataloader:
    batch_size: 256
    num_workers: 8

hooks:
  - type: CheckpointHook
    interval: 1
    max_keep: 5
  - type: LRSchedulerHook
  - type: LoggerHook
    backend: tensorboard
  - type: EarlyStoppingHook
    metric: val/accuracy
    mode: max
    patience: 10
```

## 框架模块设计

### 1. 配置解析层
- **Config Loader**：读取 YAML/JSON，支持 `base` + `override` 的多文件合并机制，提供变量插值（如 `${experiment.work_dir}`）。
- **Schema 校验**：基于 `pydantic` 或 `jsonschema` 对配置进行类型校验，避免运行时错误。

### 2. 注册表机制
- 实现 `Registry` 类，维护从字符串到类/构造函数的映射。
- 提供装饰器 `@register('resnet50', group='backbone')`，在模块导入时自动注册。
- 按功能划分子注册表：`BACKBONES`、`HEADS`、`LOSSES`、`OPTIMIZERS`、`SCHEDULERS`、`DATASETS`、`TRANSFORMS`、`HOOKS` 等。

### 3. 数据层
- **Dataset 构造器**：根据配置从注册表获取数据集类并传参实例化。
- **Transforms pipeline**：按顺序实例化数据增强操作，组合为 `Compose`。
- **Sampler & Loader**：支持分布式采样、可配置 `collate_fn`，统一封装 `build_dataloader(cfg)`。

### 4. 模型层
- **Backbone**：通过配置实例化主干网络，支持冻结部分层、载入预训练权重。
- **Head**：按配置组合多个 head（如分类、检测、分割等），并支持多损失权重叠加。
- **Model Wrapper**：封装成 `nn.Module`，提供 `forward` 与 `loss`/`predict` 等标准接口。
- **初始化模块**：根据配置执行权重初始化策略。

### 5. 训练引擎
- **Trainer**：负责迭代逻辑，包含：
  - `train_one_epoch` / `val_one_epoch`
  - 梯度累积、混合精度 (`torch.cuda.amp`)
  - 分布式训练兼容 (`DDP`/`FSDP`)
- **Optimizer & Scheduler Builder**：按配置构建优化器与学习率调度器，并支持多调度器串联。
- **钩子机制**：训练过程中在特定时刻（before/after train, epoch, iter）触发 hook。

### 6. 日志与可视化
- 统一的 `Logger` 接口，支持 TensorBoard、W&B、JSON、命令行等实现。
- 关键指标（loss、accuracy、学习率等）通过 hook 自动上报。

### 7. 检查点与恢复
- `CheckpointManager`：按配置周期保存模型权重、优化器状态、配置文件副本。
- 支持从 checkpoint 恢复训练进度，包括 epoch、迭代、随机种子。

### 8. 推理与导出
- 提供 `Inferencer`：从配置/权重加载模型并执行推理。
- 支持 `TorchScript` 或 `ONNX` 导出，导出参数也通过配置管理。

## 组件生命周期
1. **解析配置** → `cfg` 对象。
2. **构建环境**：设置随机种子、工作目录、日志、分布式环境。
3. **构建数据管线**：`build_dataloader(cfg.train_dataloader)` / `build_dataloader(cfg.val_dataloader)`。
4. **构建模型**：`build_model(cfg.model)`。
5. **构建优化器与调度器**。
6. **注册 Hook**：根据配置实例化 hook 并添加到 Trainer。
7. **开始训练**：`trainer.fit()`。
8. **评估/推理**：`trainer.evaluate()` 或 `inferencer.run()`。

## 扩展与维护建议
- 使用 `entry_points` 或模块自动发现机制加载第三方插件。
- 配置文件支持继承与参数替换，便于管理不同实验。
- 引入单元测试覆盖关键构建函数与训练循环，确保配置变化不会破坏功能。
- 提供命令行工具：`python tools/train.py --config configs/resnet50_cifar10.yaml`。
- 在文档中维护配置字段说明与示例，帮助用户快速上手。

## 目录结构示例
```
project/
├── configs/
│   ├── _base_/
│   │   ├── datasets/
│   │   ├── models/
│   │   └── schedules/
│   └── resnet50_cifar10.yaml
├── dlframework/
│   ├── __init__.py
│   ├── config/
│   ├── core/
│   ├── data/
│   ├── engine/
│   ├── hooks/
│   ├── models/
│   └── utils/
├── tools/
│   ├── train.py
│   ├── eval.py
│   └── infer.py
└── docs/
    └── user_guide.md
```

通过上述方案，可以建立一个以配置驱动的深度学习训练框架，保持高扩展性与可维护性，同时让用户无需修改代码即可快速组合不同的训练组件。
