"""Utilities for deterministic seeding."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False) -> None:
    """Set seeds across python, numpy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    """Seed dataloader workers if used."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
