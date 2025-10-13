"""Training loop and evaluation utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import torch
from pkt.config import TrainingConfig
from pkt.data import build_dataloader, build_dataset
from pkt.models import build_model
from pkt.optim import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
from pkt.utils.logging import configure_logging


class Trainer:
    """High level training orchestrator driven by :class:`TrainingConfig`."""

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        configure_logging()
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(cfg.trainer.device)
        self.output_dir = Path(cfg.trainer.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.debug("Initializing datasets and dataloaders")
        self.datasets = {
            split: build_dataset(ds_cfg.name, ds_cfg.params)
            for split, ds_cfg in cfg.datasets.items()
        }
        self.dataloaders = {
            split: build_dataloader(self.datasets[split], cfg.dataloaders[split])
            for split in self.datasets
        }

        self.logger.debug("Building model")
        self.model = build_model(cfg.model).to(self.device)

        self.logger.debug("Setting up optimizer and scheduler")
        self.optimizer = OPTIMIZER_REGISTRY.build(
            {"name": cfg.optimizer.name, "params": {**cfg.optimizer.params, "params": self.model.parameters()}}
        )
        self.scheduler = None
        if cfg.scheduler is not None:
            self.scheduler = SCHEDULER_REGISTRY.build(
                {"name": cfg.scheduler.name, "params": {**cfg.scheduler.params, "optimizer": self.optimizer}}
            )

    def train(self) -> None:
        self.logger.info("Starting training for %d epochs", self.cfg.trainer.max_epochs)
        for epoch in range(1, self.cfg.trainer.max_epochs + 1):
            train_stats = self._run_epoch(epoch, training=True)
            val_stats = None
            if "val" in self.dataloaders:
                val_stats = self._run_epoch(epoch, training=False)
            if self.scheduler is not None:
                self.scheduler.step()
            self.logger.info(
                "Epoch %d | train_loss=%.4f%s",
                epoch,
                train_stats["loss"],
                f" | val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f}" if val_stats else "",
            )

    def _run_epoch(self, epoch: int, training: bool) -> Dict[str, float]:
        split = "train" if training else "val"
        loader = self.dataloaders.get(split)
        if loader is None:
            raise KeyError(f"No dataloader configured for split '{split}'")
        if training:
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer = self.optimizer
        grad_accum = max(1, self.cfg.trainer.gradient_accumulation)
        optimizer.zero_grad(set_to_none=True)
        device = self.device

        num_batches = len(loader)
        for step, batch in enumerate(loader, 1):
            inputs, target = _to_device(batch, device)
            if training:
                output = self.model(inputs, target)
                loss = output["loss"]
                (loss / grad_accum).backward()
                if step % grad_accum == 0 or step == num_batches:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    output = self.model(inputs, target)
                    loss = output.get("loss")
                    if loss is None:
                        loss = torch.zeros((), device=device)
            running_loss += float(loss.detach().cpu())
            preds = output.get("preds")
            if preds is not None:
                correct += int((preds == target).sum().cpu())
                total += target.numel()

            if training and step % self.cfg.trainer.log_interval == 0:
                self.logger.info(
                    "Epoch %d Step %d | loss=%.4f",
                    epoch,
                    step,
                    float(loss.detach().cpu()),
                )

        avg_loss = running_loss / max(len(loader), 1)
        accuracy = correct / total if total > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}


def _to_device(batch, device: torch.device):
    if isinstance(batch, (list, tuple)):
        inputs = batch[0].to(device)
        target = batch[1].to(device)
        return inputs, target
    raise TypeError("Expected batch to be a tuple of (inputs, target)")


__all__ = ["Trainer"]
