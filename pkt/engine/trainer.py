"""Training loop and evaluation utilities."""
from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List

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
        self._start_epoch = 1
        self.global_step = 0
        self.last_checkpoint_path: Path | None = None

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
        self.save_checkpoints = bool(cfg.trainer.save_checkpoints)
        self.checkpoint_interval = max(1, cfg.trainer.checkpoint_interval)
        self.checkpoint_dir: Path | None = None
        if self.save_checkpoints:
            checkpoint_dir = Path(cfg.trainer.checkpoint_dir) if cfg.trainer.checkpoint_dir else Path("checkpoints")
            if not checkpoint_dir.is_absolute():
                checkpoint_dir = self.output_dir / checkpoint_dir
            self.checkpoint_dir = checkpoint_dir
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if cfg.trainer.resume_from:
            self.load_checkpoint(cfg.trainer.resume_from)
        elif self.checkpoint_dir:
            latest_ckpt = self.checkpoint_dir / "latest.pt"
            if latest_ckpt.exists():
                try:
                    self.load_checkpoint(latest_ckpt)
                except Exception:
                    self.logger.warning("Failed to auto-load checkpoint from %s", latest_ckpt, exc_info=True)

    def train(self) -> None:
        if self._start_epoch > self.cfg.trainer.max_epochs:
            self.logger.info(
                "Checkpoint epoch %d is beyond configured max_epochs=%d; skipping training.",
                self._start_epoch - 1,
                self.cfg.trainer.max_epochs,
            )
            return
        self.logger.info(
            "Starting training for %d epochs (resuming at epoch %d, global step %d)",
            self.cfg.trainer.max_epochs,
            self._start_epoch,
            self.global_step,
        )
        for epoch in range(self._start_epoch, self.cfg.trainer.max_epochs + 1):
            train_stats = self._run_epoch(epoch, "train", training=True)
            val_stats = None
            if "val" in self.dataloaders:
                val_stats = self._run_epoch(epoch, "val", training=False)
            if self.scheduler is not None:
                self.scheduler.step()
            self.logger.info(
                "Epoch %d | train_loss=%.4f%s",
                epoch,
                train_stats["loss"],
                f" | val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f}" if val_stats else "",
            )
            self._save_checkpoint(epoch, train_stats, val_stats)

    def evaluate(self, split: str = "val") -> Dict[str, float]:
        """Run evaluation on the specified split without updating model parameters."""

        if split not in self.dataloaders:
            raise KeyError(f"No dataloader configured for split '{split}'")
        # Use 0 to indicate evaluation-only run in logs.
        stats = self._run_epoch(epoch=max(self._start_epoch - 1, 0), split=split, training=False)
        self.logger.info(
            "Eval %s | loss=%.4f accuracy=%.4f",
            split,
            stats["loss"],
            stats["accuracy"],
        )
        return stats

    def predict(
        self,
        split: str = "val",
        *,
        return_logits: bool = False,
        return_targets: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run inference on a split and return concatenated predictions."""

        loader = self.dataloaders.get(split)
        if loader is None:
            raise KeyError(f"No dataloader configured for split '{split}'")
        if self.cfg.dataloaders[split].shuffle:
            self.logger.warning("Dataloader for split '%s' has shuffle=True; prediction order may be nondeterministic.", split)

        self.model.eval()
        preds: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []
        targets_list: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                inputs, target = _to_device(batch, self.device)
                output = self.model(inputs)
                logits = output.get("logits")
                if logits is None:
                    raise KeyError("Model output does not contain 'logits'; cannot compute predictions.")
                pred_batch = output.get("preds")
                if pred_batch is None:
                    pred_batch = torch.argmax(logits, dim=1)
                preds.append(pred_batch.cpu())
                if return_logits:
                    logits_list.append(logits.cpu())
                if return_targets:
                    if target is None:
                        raise ValueError("Requested targets for split '%s' but dataset does not provide them.", split)
                    targets_list.append(target.cpu())

        result: Dict[str, torch.Tensor] = {"preds": torch.cat(preds) if preds else torch.empty(0, dtype=torch.long)}
        if return_logits:
            result["logits"] = torch.cat(logits_list) if logits_list else torch.empty((0,), dtype=torch.float32)
        if return_targets:
            result["targets"] = torch.cat(targets_list) if targets_list else torch.empty(0, dtype=torch.long)
        return result

    def _run_epoch(self, epoch: int, split: str, training: bool) -> Dict[str, float]:
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
            if training and target is None:
                raise ValueError(f"Training requires targets but split '{split}' yielded none.")
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
            if preds is not None and target is not None:
                correct += int((preds == target).sum().cpu())
                total += target.numel()

            if training:
                self.global_step += 1
            if training and step % self.cfg.trainer.log_interval == 0:
                self.logger.info(
                    "Epoch %d Step %d (global %d) | loss=%.4f",
                    epoch,
                    step,
                    self.global_step,
                    float(loss.detach().cpu()),
                )

        avg_loss = running_loss / max(len(loader), 1)
        accuracy = correct / total if total > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

    def _checkpoint_path(self, epoch: int) -> Path:
        if not self.checkpoint_dir:
            raise RuntimeError("Checkpoint directory is not configured")
        return self.checkpoint_dir / f"epoch_{epoch:04d}.pt"

    def _save_checkpoint(
        self,
        epoch: int,
        train_stats: Dict[str, float],
        val_stats: Dict[str, float] | None,
    ) -> None:
        if not self.save_checkpoints or not self.checkpoint_dir:
            return
        if epoch % self.checkpoint_interval != 0 and epoch != self.cfg.trainer.max_epochs:
            return
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "train_metrics": train_stats,
            "val_metrics": val_stats,
            "cfg": self.cfg.as_dict(),
            "rng_state": self._capture_rng_state(),
        }
        path = self._checkpoint_path(epoch)
        torch.save(checkpoint, path)
        self.last_checkpoint_path = path

        latest_path = self.checkpoint_dir / "latest.pt"
        try:
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(path)
        except (OSError, NotImplementedError):
            try:
                shutil.copy2(path, latest_path)
            except OSError:
                self.logger.warning("Failed to update latest checkpoint at %s", latest_path, exc_info=True)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        self.logger.info("Loading checkpoint from %s", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model_state = checkpoint.get("model")
        if model_state is None:
            raise KeyError("Checkpoint is missing 'model' state_dict")
        self.model.load_state_dict(model_state)

        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        scheduler_state = checkpoint.get("scheduler")
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        self._start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self._restore_rng_state(checkpoint.get("rng_state"))

        if self.save_checkpoints and self.checkpoint_dir and ckpt_path.parent != self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint_path = ckpt_path

    def _capture_rng_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "random": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                state["cuda"] = torch.cuda.get_rng_state_all()
            except RuntimeError:
                self.logger.debug("CUDA RNG state unavailable despite CUDA being reported available.")
        try:
            import numpy as np  # type: ignore
        except ModuleNotFoundError:
            state["numpy"] = None
        else:
            state["numpy"] = np.random.get_state()
        return state

    def _restore_rng_state(self, state: Dict[str, Any] | None) -> None:
        if not state:
            return
        random_state = state.get("random")
        if random_state is not None:
            random.setstate(random_state)
        torch_state = state.get("torch")
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        numpy_state = state.get("numpy")
        if numpy_state is not None:
            try:
                import numpy as np  # type: ignore
            except ModuleNotFoundError:
                self.logger.warning("NumPy not available; cannot restore RNG state from checkpoint.")
            else:
                np.random.set_state(numpy_state)
        cuda_state = state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_state)
            except RuntimeError:
                self.logger.warning("Failed to restore CUDA RNG state from checkpoint.", exc_info=True)


def _to_device(batch, device: torch.device):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            inputs = batch[0].to(device)
            target_tensor = batch[1]
            if target_tensor is not None:
                target_tensor = target_tensor.to(device)
            return inputs, target_tensor
        if len(batch) == 1:
            inputs = batch[0].to(device)
            return inputs, None
    elif isinstance(batch, dict):
        if "inputs" not in batch:
            raise KeyError("Dictionary batch must contain an 'inputs' key")
        inputs = batch["inputs"].to(device)
        target_tensor = batch.get("targets") or batch.get("target")
        if target_tensor is not None:
            target_tensor = target_tensor.to(device)
        return inputs, target_tensor
    elif torch.is_tensor(batch):
        return batch.to(device), None
    raise TypeError("Unsupported batch format for moving to device")


__all__ = ["Trainer"]
