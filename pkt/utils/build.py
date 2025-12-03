"""Helpers for instantiating objects from configuration dictionaries."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, MutableMapping

from pkt.registry import Registry


def build_from_cfg(
    cfg: MutableMapping[str, Any] | Mapping[str, Any] | None,
    registry: Registry,
    default_args: Mapping[str, Any] | None = None,
) -> Any:
    if cfg is None:
        return None
    if not isinstance(cfg, MutableMapping):
        if isinstance(cfg, Mapping):
            cfg = dict(cfg)
        else:
            raise TypeError(f"Config must be a mapping, got {type(cfg)!r}")
    cfg = deepcopy(cfg)
    obj_type = cfg.pop("type", None) or cfg.pop("name", None)
    if obj_type is None:
        raise KeyError(f"Configuration missing 'type' (or 'name'): {cfg}")
    try:
        obj_cls = registry.get(obj_type)
    except KeyError as exc:
        raise KeyError(f"Failed to build '{obj_type}' from registry '{registry._name}'") from exc
    if default_args:
        merged = dict(default_args)
        merged.update(cfg)
        cfg = merged
    return obj_cls(**cfg)


__all__ = ["build_from_cfg"]
