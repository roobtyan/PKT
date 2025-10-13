"""Factory helpers for datasets and dataloaders."""
from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from pkt.config import DataLoaderConfig


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


__all__ = ["build_dataloader"]
