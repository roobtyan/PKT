"""Central registry utilities for the configuration-driven framework."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, Optional


class Registry:
    """A lightweight registry used to map string keys to callables or classes."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._module_dict: Dict[str, Callable[..., Any]] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    def __iter__(self) -> Iterator[str]:
        return iter(self._module_dict)

    def register(self, name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a callable or class.

        Parameters
        ----------
        name:
            The key under which the callable will be registered. If ``None`` the
            callable's ``__name__`` is used.
        """

        def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
            key = name or obj.__name__
            if key in self._module_dict:
                raise KeyError(f"{key!r} is already registered in {self._name} registry")
            self._module_dict[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._module_dict[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._module_dict)) or "<empty>"
            raise KeyError(
                f"{name!r} is not registered in {self._name} registry. "
                f"Available: {available}"
            ) from exc

    def build(self, cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Instantiate a registered object from a configuration mapping."""
        if "name" not in cfg:
            raise KeyError(f"Configuration for {self._name} registry requires a 'name' key: {cfg}")
        builder = self.get(cfg["name"])
        params = cfg.get("params", {}) or {}
        if not isinstance(params, dict):
            raise TypeError(
                f"Expected 'params' for {self._name}:{cfg['name']} to be a mapping, got {type(params)!r}"
            )
        return builder(*args, **params, **kwargs)

    def items(self) -> Iterator[tuple[str, Callable[..., Any]]]:
        return self._module_dict.items()

    def keys(self) -> Iterator[str]:
        return self._module_dict.keys()

    def values(self) -> Iterator[Callable[..., Any]]:
        return self._module_dict.values()


__all__ = ["Registry"]
