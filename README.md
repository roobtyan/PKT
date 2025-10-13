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
