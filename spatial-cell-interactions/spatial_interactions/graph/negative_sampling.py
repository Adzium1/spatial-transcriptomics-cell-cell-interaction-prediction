"""Negative edge sampling helpers."""

from __future__ import annotations

import torch
from torch_geometric.utils import negative_sampling as pyg_negative_sampling


def sample_negative_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_samples: int | None = None,
    ratio: float = 1.0,
    method: str = "sparse",
) -> torch.Tensor:
    """
    Sample non-edge pairs for self-supervised training.
    """
    if num_samples is None:
        num_samples = int(edge_index.size(1) * ratio)
    neg_edges = pyg_negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_samples,
        method=method,
    )
    return neg_edges
