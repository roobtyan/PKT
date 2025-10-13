"""Logging utilities."""
from __future__ import annotations

import logging
import sys


_LOGGING_CONFIGURED = False


def configure_logging(level: int = logging.INFO, force: bool = False) -> None:
    """Configure basic logging for the framework."""

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED and not force:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    _LOGGING_CONFIGURED = True


__all__ = ["configure_logging"]
