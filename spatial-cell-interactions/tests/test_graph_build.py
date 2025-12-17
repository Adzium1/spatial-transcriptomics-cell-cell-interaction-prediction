from pathlib import Path
import sys

import numpy as np
import anndata as ad

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from spatial_interactions.graph.build_graph import build_spatial_graph


def test_knn_graph_edge_count():
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    X = np.random.rand(4, 3)
    adata = ad.AnnData(X=X)
    adata.obsm["spatial"] = coords

    artifacts = build_spatial_graph(adata, k=2, rbf_dim=4)
    edge_index = artifacts.data.edge_index
    assert edge_index.shape[1] == coords.shape[0] * 2
    assert np.all(artifacts.distances >= 0)
