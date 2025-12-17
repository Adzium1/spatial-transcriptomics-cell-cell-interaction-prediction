"""Feature selection utilities."""

from __future__ import annotations

import scanpy as sc

from spatial_interactions.utils.logging import get_logger

logger = get_logger(__name__)


def select_hvgs(
    adata: "sc.AnnData",
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    subset: bool = True,
) -> "sc.AnnData":
    """Compute and optionally subset to highly variable genes."""
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor, subset=False)
    hvgs = adata.var.index[adata.var["highly_variable"]]
    logger.info("Selected %d highly variable genes using %s", len(hvgs), flavor)
    if subset:
        adata = adata[:, hvgs].copy()
    return adata
