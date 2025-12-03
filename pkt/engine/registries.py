"""Shared registries for the configurable training/run-time framework."""
from __future__ import annotations

from pkt.registry import Registry


MODULES = Registry("module")
PIPELINES = Registry("pipeline")
VISUALIZERS = Registry("visualizer")
HOOKS = Registry("hook")
RUNNERS = Registry("runner")
COLLATE_FUNCTIONS = Registry("collate")


__all__ = [
    "MODULES",
    "PIPELINES",
    "VISUALIZERS",
    "HOOKS",
    "RUNNERS",
    "COLLATE_FUNCTIONS",
]
