"""Loss utilities."""

from __future__ import annotations

import torch
from torch import nn


def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with logits."""
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)
