"""Lightweight runner implementations for configuration-driven workflows."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch

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

    def register_hook(self, hook: Hook) -> None:
        self.hooks.append(hook)
        self.hooks.sort(key=lambda h: getattr(h, "priority", 50))

    def call_hook(self, method: str, *args, **kwargs) -> None:
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
        datasets_cfg = cfg.get("datasets", {})
        if "train" not in datasets_cfg:
            raise KeyError("PipelineRunner requires a 'train' dataset configuration")
        self.dataset = build_dataset_from_cfg(datasets_cfg["train"])

        dataloader_cfg = cfg.get("dataloaders", {}).get("train", {})
        self.dataloader = build_dataloader_from_cfg(self.dataset, dataloader_cfg)

        model_cfg = cfg.get("model")
        if model_cfg is None:
            raise KeyError("PipelineRunner requires a 'model' configuration")
        self.model = build_from_cfg(model_cfg, MODULES)
        self.model.to(self.device)
        self.model.eval()

        hook_cfgs = cfg.get("hooks", [])
        for hook_cfg in hook_cfgs:
            hook = build_from_cfg(hook_cfg, HOOKS)
            self.register_hook(hook)

    def run(self) -> None:
        self.call_hook("before_run")
        iteration = 0
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


def build_runner(cfg: Dict[str, Any]) -> BaseRunner:
    runner_cfg = cfg.get("runner")
    if runner_cfg is None:
        raise KeyError("Configuration missing 'runner' section")
    runner_spec = dict(runner_cfg)
    return build_from_cfg(runner_spec, RUNNERS, default_args={"cfg": cfg})


__all__ = ["PipelineRunner", "build_runner"]
