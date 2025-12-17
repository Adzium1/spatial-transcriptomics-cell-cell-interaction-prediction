"""Lightweight logging helpers."""

from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str = "spatial_interactions", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a module logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
