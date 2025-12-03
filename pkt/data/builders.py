"""Factory helpers for datasets and dataloaders."""
from __future__ import annotations

from functools import partial
from typing import Any, Mapping

from torch.utils.data import DataLoader, Dataset

from pkt.config import DataLoaderConfig
from pkt.data.datasets import DATASET_REGISTRY
from pkt.engine.registries import COLLATE_FUNCTIONS
from pkt.utils.build import build_from_cfg


def build_dataloader(dataset: Dataset, cfg: DataLoaderConfig) -> DataLoader:
    """Create a :class:`torch.utils.data.DataLoader` from configuration."""

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        drop_last=cfg.drop_last,
        pin_memory=cfg.pin_memory,
        **cfg.additional_args,
    )


def build_dataset_from_cfg(cfg: Mapping[str, Any]):
    cfg = dict(cfg)
    obj_type = cfg.pop("type", None)
    if obj_type is None:
        raise KeyError("Dataset config must include 'type'")
    builder = DATASET_REGISTRY.get(obj_type)
    return builder(**cfg)


def build_dataloader_from_cfg(dataset: Dataset, cfg: Mapping[str, Any]) -> DataLoader:
    cfg = dict(cfg)
    collate_cfg = cfg.pop("collate_fn", None)
    collate_fn = None
    if collate_cfg is not None:
        if not isinstance(collate_cfg, Mapping):
            raise TypeError("collate_fn configuration must be a mapping")
        collate_cfg = dict(collate_cfg)
        func_type = collate_cfg.pop("type", None) or collate_cfg.pop("name", None)
        if func_type is None:
            raise KeyError("collate_fn configuration requires a 'type' or 'name'")
        func = COLLATE_FUNCTIONS.get(func_type)
        collate_fn = partial(func, **collate_cfg) if collate_cfg else func
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=bool(cfg.get("shuffle", False)),
        num_workers=int(cfg.get("num_workers", 0)),
        drop_last=bool(cfg.get("drop_last", False)),
        pin_memory=bool(cfg.get("pin_memory", False)),
        collate_fn=collate_fn,
    )


__all__ = ["build_dataloader", "build_dataset_from_cfg", "build_dataloader_from_cfg"]
