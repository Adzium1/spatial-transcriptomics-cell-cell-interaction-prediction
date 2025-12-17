"""Edge scoring head for self-supervised reconstruction."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class EdgeScoreHead(nn.Module):
    """MLP over concatenated node embeddings and edge attributes."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dims: Iterable[int] = (128, 64)) -> None:
        super().__init__()
        dims: List[int] = [in_dim * 2 + edge_dim] + list(hidden_dims) + [1]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(0.1)])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        feat = torch.cat([z[src], z[dst], edge_attr], dim=-1)
        logits = self.net(feat).squeeze(-1)
        return logits
