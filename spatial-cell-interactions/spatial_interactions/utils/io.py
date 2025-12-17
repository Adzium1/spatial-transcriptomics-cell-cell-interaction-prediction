"""I/O helpers for configs and filesystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    """Save YAML configuration file."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save JSON with pretty formatting."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
