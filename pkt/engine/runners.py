"""Lightweight runner implementations for configuration-driven workflows."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from pkt.data.builders import build_dataset_from_cfg, build_dataloader_from_cfg
from pkt.engine.hooks import Hook
from pkt.engine.registries import HOOKS, MODULES, RUNNERS
from pkt.utils.build import build_from_cfg
import pkt.models  # ensure module registries populated


def _to_device(inputs: Any, device: torch.device) -> Any:
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    if isinstance(inputs, Mapping):
        return {k: _to_device(v, device) for k, v in inputs.items()}
    if isinstance(inputs, list):
        return [_to_device(v, device) for v in inputs]
    if isinstance(inputs, tuple):
        return tuple(_to_device(v, device) for v in inputs)
    return inputs


class BaseRunner:
    def __init__(
        self,
        cfg: Dict[str, Any],
        device: str | None = None,
        max_iters: int | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.cfg = cfg
        runner_cfg = cfg.get("runner", {})
        runner_device = device or runner_cfg.get("device", "cpu")
        runner_iters = max_iters or runner_cfg.get("max_iters", 1)
        self.device = torch.device(runner_device)
        self.max_iters = int(runner_iters)
        self.hooks: List[Hook] = []
        self.is_main_process = True

    def register_hook(self, hook: Hook) -> None:
        self.hooks.append(hook)
        self.hooks.sort(key=lambda h: getattr(h, "priority", 50))

    def call_hook(self, method: str, *args, **kwargs) -> None:
        if not self.is_main_process:
            return
        for hook in self.hooks:
            fn = getattr(hook, method, None)
            if callable(fn):
                fn(self, *args, **kwargs)

    def run(self) -> None:
        raise NotImplementedError


@RUNNERS.register("PipelineRunner")
class PipelineRunner(BaseRunner):
    def __init__(self, cfg: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        runner_cfg = cfg.get("runner", {})
        dist_cfg = runner_cfg.get("distributed", {})
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self._setup_distributed(dist_cfg)

        datasets_cfg = cfg.get("datasets", {})
        if "train" not in datasets_cfg:
            raise KeyError("PipelineRunner requires a 'train' dataset configuration")
        self.dataset = build_dataset_from_cfg(datasets_cfg["train"])

        dataloader_cfg = cfg.get("dataloaders", {}).get("train", {})
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=bool(dataloader_cfg.get("shuffle", False)),
                drop_last=bool(dataloader_cfg.get("drop_last", False)),
            )
        self.dataloader = build_dataloader_from_cfg(self.dataset, dataloader_cfg, sampler=sampler)
        self.sampler = sampler

        model_cfg = cfg.get("model")
        if model_cfg is None:
            raise KeyError("PipelineRunner requires a 'model' configuration")
        self.model = build_from_cfg(model_cfg, MODULES)
        self.model.to(self.device)
        if self.is_distributed:
            device_ids = None
            if self.device.type == "cuda":
                device_ids = [self.device.index]
            self.model = DDP(
                self.model,
                device_ids=device_ids,
                output_device=self.device if device_ids else None,
                find_unused_parameters=bool(dist_cfg.get("find_unused_parameters", False)),
            )
        self.model.eval()

        hook_cfgs = cfg.get("hooks", [])
        for hook_cfg in hook_cfgs:
            hook = build_from_cfg(hook_cfg, HOOKS)
            self.register_hook(hook)

    def run(self) -> None:
        self.call_hook("before_run")
        iteration = 0
        if self.is_distributed and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(0)
        with torch.no_grad():
            for batch in self.dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, _ = batch
                else:
                    inputs = batch
                inputs = _to_device(inputs, self.device)
                outputs = self.model(inputs)
                if not isinstance(outputs, Mapping):
                    raise TypeError("Model output must be a mapping for PipelineRunner")
                iteration += 1
                self.call_hook("after_iter", outputs, iteration)
                if iteration >= self.max_iters:
                    break
        self.call_hook("after_run")
        if self.is_distributed:
            dist.barrier()
            dist.destroy_process_group()

    def _setup_distributed(self, dist_cfg: Mapping[str, Any]) -> None:
        if not dist_cfg or not dist_cfg.get("enabled", False):
            return
        if self.device.type == "mps":
            raise ValueError("Distributed training is not supported on MPS devices.")
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available but distributed was requested.")
        backend = dist_cfg.get("backend") or ("nccl" if torch.cuda.is_available() else "gloo")
        init_method = dist_cfg.get("init_method", "env://")
        init_kwargs = {"backend": backend, "init_method": init_method}
        if "world_size" in dist_cfg:
            init_kwargs["world_size"] = int(dist_cfg["world_size"])
        if "rank" in dist_cfg:
            init_kwargs["rank"] = int(dist_cfg["rank"])
        dist.init_process_group(**init_kwargs)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if self.device.type == "cuda":
            local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
        self.is_main_process = self.rank == 0
        self.is_distributed = True


def build_runner(cfg: Dict[str, Any]) -> BaseRunner:
    runner_cfg = cfg.get("runner")
    if runner_cfg is None:
        raise KeyError("Configuration missing 'runner' section")
    runner_spec = dict(runner_cfg)
    return build_from_cfg(runner_spec, RUNNERS, default_args={"cfg": cfg})


__all__ = ["PipelineRunner", "build_runner"]
