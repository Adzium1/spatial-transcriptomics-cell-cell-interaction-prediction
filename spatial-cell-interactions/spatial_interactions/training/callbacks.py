"""Training callbacks such as early stopping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from spatial_interactions.utils.io import ensure_dir


def _to_cpu(obj):
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if hasattr(obj, "cpu"):
        return obj.cpu()
    return obj


@dataclass
class EarlyStopping:
    monitor: str = "val_ap"
    patience: int = 10
    mode: str = "max"
    best_score: float = float("-inf")
    counter: int = 0
    best_state: Optional[dict] = None

    def step(self, score: float, state_dict: dict) -> bool:
        improved = (score > self.best_score) if self.mode == "max" else (score < self.best_score)
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_state = _to_cpu(state_dict)
        else:
            self.counter += 1
        return self.counter >= self.patience

    def save_best(self, path: Path, model: torch.nn.Module) -> None:
        if self.best_state is None:
            return
        ensure_dir(path.parent)
        torch.save({"model_state": self.best_state}, path)
