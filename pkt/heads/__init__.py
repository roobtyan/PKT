"""Head registry与基础定义。"""
from __future__ import annotations

from pkt.registry import Registry

HEAD_REGISTRY = Registry("head")

# 导入子模块以完成注册
from pkt.heads.base_head import BaseHead
from pkt.heads.classification import ClassificationHead

__all__ = ["HEAD_REGISTRY", "BaseHead", "ClassificationHead"]
