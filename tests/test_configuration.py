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


def test_checkpoint_save_and_resume(tmp_path: Path):
    cfg = load_yaml_config(Path("configs/random_classification.yaml"))
    cfg.trainer.output_dir = str(tmp_path / "outputs")
    cfg.trainer.max_epochs = 2
    cfg.trainer.checkpoint_interval = 1
    trainer = Trainer(cfg)
    trainer.train()
    ckpt_dir = Path(cfg.trainer.output_dir) / "checkpoints"
    ckpt_file = ckpt_dir / "epoch_0002.pt"
    latest_file = ckpt_dir / "latest.pt"
    assert ckpt_file.exists()
    assert latest_file.exists()
    initial_global_step = trainer.global_step
    assert initial_global_step > 0

    resume_cfg = load_yaml_config(Path("configs/random_classification.yaml"))
    resume_cfg.trainer.output_dir = cfg.trainer.output_dir
    resume_cfg.trainer.max_epochs = 3
    resume_cfg.trainer.resume_from = str(ckpt_file)
    resume_cfg.trainer.checkpoint_interval = 1
    resume_trainer = Trainer(resume_cfg)

    assert resume_trainer._start_epoch == 3
    assert resume_trainer.global_step == initial_global_step
    resume_trainer.train()
    assert resume_trainer.global_step > initial_global_step
    assert (Path(resume_cfg.trainer.output_dir) / "checkpoints" / "epoch_0003.pt").exists()


def test_trainer_evaluate_and_predict(tmp_path: Path):
    cfg = load_yaml_config(Path("configs/random_classification.yaml"))
    cfg.trainer.output_dir = str(tmp_path / "outputs")
    cfg.trainer.max_epochs = 1
    trainer = Trainer(cfg)

    metrics = trainer.evaluate(split="val")
    assert "loss" in metrics and "accuracy" in metrics
    assert trainer.global_step == 0

    predictions = trainer.predict(split="val", return_logits=True, return_targets=True)
    val_len = len(trainer.datasets["val"])
    assert predictions["preds"].shape[0] == val_len
    assert predictions["logits"].shape[0] == val_len
    assert predictions["targets"].shape == predictions["preds"].shape
