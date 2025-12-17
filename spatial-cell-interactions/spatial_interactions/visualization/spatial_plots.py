"""Spatial plots with predicted interaction edges."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from spatial_interactions.utils.io import ensure_dir


def plot_spatial_edges(
    coords: np.ndarray,
    edge_index: np.ndarray,
    edge_scores: np.ndarray,
    out_path: Path,
    top_k: int = 500,
    point_size: float = 8.0,
) -> None:
    """Plot tissue coordinates and overlay top-k edges by score."""
    ensure_dir(out_path.parent)
    order = np.argsort(edge_scores)[::-1]
    top = order[: min(top_k, edge_index.shape[1])]
    edges = edge_index[:, top]
    scores = edge_scores[top]
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    plt.figure(figsize=(8, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=point_size, c="lightgray", alpha=0.6, edgecolors="none")
    for idx, (src, dst) in enumerate(edges.T):
        c = plt.cm.magma(norm_scores[idx])
        plt.plot([coords[src, 0], coords[dst, 0]], [coords[src, 1], coords[dst, 1]], color=c, alpha=0.4, linewidth=1.0)
    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.title(f"Top {len(top)} predicted interaction edges")
    sm = plt.cm.ScalarMappable(cmap="magma")
    sm.set_array(scores)
    plt.colorbar(sm, fraction=0.046, pad=0.04, label="Edge score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
