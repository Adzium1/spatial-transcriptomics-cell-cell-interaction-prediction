"""UMAP embedding visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from spatial_interactions.utils.io import ensure_dir


def plot_embeddings_umap(
    embeddings: np.ndarray,
    obs: Optional[pd.DataFrame],
    out_path: Path,
    color_by: Optional[str] = None,
    n_neighbors: int = 15,
) -> None:
    """Compute UMAP on embeddings and save scatter plot."""
    ensure_dir(out_path.parent)
    ad = sc.AnnData(embeddings)
    if obs is not None:
        ad.obs = obs.copy()
    sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.umap(ad)
    if color_by and color_by in ad.obs.columns:
        color = ad.obs[color_by]
        title = f"UMAP colored by {color_by}"
    else:
        sc.tl.leiden(ad, key_added="leiden")
        color = ad.obs["leiden"]
        title = "UMAP colored by Leiden clusters"

    coords = ad.obsm["X_umap"]
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=color.astype("category").cat.codes if hasattr(color, "cat") else color, cmap="viridis", s=10, alpha=0.8)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(title)
    if color_by:
        plt.colorbar(scatter, label=color_by)
    else:
        plt.colorbar(scatter, label="Leiden")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
