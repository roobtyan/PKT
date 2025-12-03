"""Hook utilities for custom runners."""
from __future__ import annotations

from typing import Any, Dict, List

from pkt.engine.registries import HOOKS, VISUALIZERS
from pkt.utils.build import build_from_cfg
import pkt.visualize  # ensure visualizers register themselves


class Hook:
    priority: int = 50

    def before_run(self, runner: "BaseRunner") -> None:  # noqa: F821
        pass

    def after_run(self, runner: "BaseRunner") -> None:  # noqa: F821
        pass

    def after_iter(self, runner: "BaseRunner", outputs: Dict[str, Any], iteration: int) -> None:
        pass


@HOOKS.register("VisualizationHook")
class VisualizationHook(Hook):
    def __init__(
        self,
        visualizers: List[Dict[str, Any]],
        interval: int = 1,
    ) -> None:
        if not visualizers:
            raise ValueError("VisualizationHook requires at least one visualizer configuration")
        self.interval = max(1, interval)
        self.visualizers = [build_from_cfg(cfg, VISUALIZERS) for cfg in visualizers]

    def after_iter(self, runner: "BaseRunner", outputs: Dict[str, Any], iteration: int) -> None:
        if iteration % self.interval != 0:
            return
        data = outputs
        for visualizer in self.visualizers:
            visualizer(data)


__all__ = ["Hook", "VisualizationHook"]
