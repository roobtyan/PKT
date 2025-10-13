from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pkt import Trainer, load_yaml_config


def test_load_and_build(tmp_path: Path):
    cfg_path = Path("configs/random_classification.yaml")
    cfg = load_yaml_config(cfg_path)
    assert cfg.datasets["train"].name == "random_classification"
    trainer = Trainer(cfg)
    # ensure forward pass works for one batch
    train_loader = trainer.dataloaders["train"]
    batch = next(iter(train_loader))
    inputs, targets = batch
    outputs = trainer.model(inputs, targets)
    assert "loss" in outputs
    assert outputs["logits"].shape[0] == inputs.shape[0]
    assert torch.isfinite(outputs["loss"]).all()
