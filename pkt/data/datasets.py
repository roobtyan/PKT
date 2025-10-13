"""Dataset implementations and builders."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Dataset

from pkt.registry import Registry


DATASET_REGISTRY = Registry("dataset")


class BaseDataset(Dataset):
    """Base dataset with optional common utilities."""

    def __init__(self) -> None:
        super().__init__()


@DATASET_REGISTRY.register("random_classification")
class RandomClassificationDataset(BaseDataset):
    """Randomly generated samples for quick experiments and tests."""

    def __init__(
        self,
        num_samples: int,
        num_features: int,
        num_classes: int,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        features = torch.randn(num_samples, num_features, generator=generator)
        labels = torch.randint(low=0, high=num_classes, size=(num_samples,), generator=generator)
        self.features = features
        self.labels = labels

    def __len__(self) -> int:  # type: ignore[override]
        return self.features.size(0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.features[index], self.labels[index]


def build_dataset(name: str, params: Dict[str, Any]) -> Dataset:
    """Build a dataset from the :mod:`DATASET_REGISTRY`."""

    builder = DATASET_REGISTRY.get(name)
    return builder(**params)


__all__ = [
    "BaseDataset",
    "RandomClassificationDataset",
    "DATASET_REGISTRY",
    "build_dataset",
]
