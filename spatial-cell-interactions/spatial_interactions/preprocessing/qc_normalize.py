"""Quality control and normalization routines."""

from __future__ import annotations

from typing import Optional

import numpy as np
import scanpy as sc

from spatial_interactions.utils.logging import get_logger

logger = get_logger(__name__)


def filter_spots(adata: "sc.AnnData", keep_all_if_missing: bool = True) -> "sc.AnnData":
    """Filter to in-tissue spots when metadata is present."""
    if "in_tissue" in adata.obs.columns:
        before = adata.n_obs
        adata = adata[adata.obs["in_tissue"] == 1].copy()
        logger.info("Filtered spots: %d -> %d in-tissue", before, adata.n_obs)
    elif keep_all_if_missing:
        logger.warning("No 'in_tissue' column found; keeping all spots.")
    else:
        raise ValueError("in_tissue metadata missing and keep_all_if_missing is False.")
    return adata


def filter_genes_by_pct(
    adata: "sc.AnnData", min_pct: float = 0.01, inplace: bool = False
) -> "sc.AnnData":
    """Remove genes expressed in fewer than min_pct of spots."""
    min_cells = int(np.ceil(min_pct * adata.n_obs))
    sc.pp.filter_genes(adata, min_cells=min_cells, inplace=True)
    logger.info("Filtered genes with min_pct=%.3f (min_cells=%d)", min_pct, min_cells)
    return adata if inplace else adata.copy()


def normalize_log1p(adata: "sc.AnnData") -> "sc.AnnData":
    """Total-count normalize to 1e4 then log1p."""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = adata.X.astype(np.float32)
    return adata
