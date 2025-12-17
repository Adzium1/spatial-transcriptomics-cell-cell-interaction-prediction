"""Ligand-receptor scoring utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from spatial_interactions.utils.logging import get_logger

logger = get_logger(__name__)


def rank_ligand_receptor_pairs(
    adata: "anndata.AnnData",
    edge_index: np.ndarray,
    edge_scores: np.ndarray,
    lr_csv: Path,
    top_edges: int = 500,
) -> Optional[pd.DataFrame]:
    """
    Compute ligand-receptor pair enrichment among top predicted edges.
    """
    try:
        import anndata  # noqa: F401
    except ImportError as exc:
        raise ImportError("anndata is required for ligand-receptor analysis") from exc

    if not lr_csv.exists():
        logger.warning("Ligand-receptor file %s not found; skipping LR analysis.", lr_csv)
        return None
    lr_df = pd.read_csv(lr_csv)
    required_cols = {"ligand", "receptor"}
    if not required_cols.issubset(set(lr_df.columns)):
        raise ValueError(f"Ligand-receptor CSV must contain columns {required_cols}")

    order = np.argsort(edge_scores)[::-1]
    top_idx = order[: min(top_edges, edge_index.shape[1])]
    edges = edge_index[:, top_idx]
    scores = edge_scores[top_idx]

    results = []
    expr = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    for _, row in lr_df.iterrows():
        lig, rec = row["ligand"], row["receptor"]
        if lig not in gene_to_idx or rec not in gene_to_idx:
            continue
        lig_exp = expr[:, gene_to_idx[lig]]
        rec_exp = expr[:, gene_to_idx[rec]]
        lr_edge_scores = lig_exp[edges[0]] * rec_exp[edges[1]]
        if lr_edge_scores.size == 0:
            continue
        corr, _ = spearmanr(scores, lr_edge_scores)
        results.append(
            {
                "ligand": lig,
                "receptor": rec,
                "mean_lr_edge_score": float(np.mean(lr_edge_scores)),
                "spearman_with_pred": float(corr) if corr == corr else float("nan"),
            }
        )

    if not results:
        logger.warning("No ligand-receptor pairs matched genes in the dataset.")
        return None
    df = pd.DataFrame(results).sort_values("mean_lr_edge_score", ascending=False)
    return df
