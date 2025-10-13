"""Optimization registries."""
from __future__ import annotations

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from pkt.registry import Registry


OPTIMIZER_REGISTRY = Registry("optimizer")
SCHEDULER_REGISTRY = Registry("scheduler")


@OPTIMIZER_REGISTRY.register("sgd")
def build_sgd(params, lr: float, momentum: float = 0.0, weight_decay: float = 0.0):
    return optim.SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay)


@OPTIMIZER_REGISTRY.register("adam")
def build_adam(params, lr: float, weight_decay: float = 0.0, betas=(0.9, 0.999)):
    return optim.Adam(params=params, lr=lr, weight_decay=weight_decay, betas=betas)


@SCHEDULER_REGISTRY.register("step_lr")
def build_step_lr(optimizer, step_size: int, gamma: float = 0.1):
    return StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)


__all__ = [
    "OPTIMIZER_REGISTRY",
    "SCHEDULER_REGISTRY",
    "build_sgd",
    "build_adam",
    "build_step_lr",
]
