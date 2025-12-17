"""Distance-aware GATv2 encoder."""

from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


class DistanceAwareGATv2(nn.Module):
    """
    GATv2 encoder that injects edge attributes into attention.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        edge_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[GATv2Conv] = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            in_channels = dims[i]
            out_channels = dims[i + 1]
            # keep overall embedding size stable by dividing with heads for intermediate layers
            concat = i != num_layers - 1
            conv = GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels // heads if concat else out_channels,
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout,
                add_self_loops=False,
                concat=concat,
            )
            layers.append(conv)
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index=edge_index, edge_attr=edge_attr)
        return x
