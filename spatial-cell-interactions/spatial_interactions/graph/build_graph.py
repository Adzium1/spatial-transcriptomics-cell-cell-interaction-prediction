"""Build spatial neighbor graphs with distance encodings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from spatial_interactions.utils.io import ensure_dir
from spatial_interactions.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GraphArtifacts:
    data: Data
    distances: np.ndarray
    rbf_centers: np.ndarray


def radial_basis_encoding(
    distances: np.ndarray,
    num_centers: int = 16,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand scalar distances with Gaussian RBFs.

    Returns encoding matrix and the centers used.
    """
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        raise ValueError("No finite distances provided for RBF encoding.")
    d_min = float(np.min(finite)) if d_min is None else d_min
    d_max = float(np.max(finite)) if d_max is None else d_max
    if d_max - d_min < epsilon:
        d_max = d_min + epsilon

    centers = np.linspace(d_min, d_max, num_centers)
    width = (centers[1] - centers[0]) if num_centers > 1 else 1.0
    gamma = 1.0 / (2 * (width**2))

    dist = distances[:, None]
    enc = np.exp(-gamma * (dist - centers[None, :]) ** 2)
    return enc.astype(np.float32), centers


def _knn_edges(coords: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto").fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    # remove self neighbors (distance=0)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    sources = np.repeat(np.arange(coords.shape[0]), k)
    targets = indices.reshape(-1)
    dists = distances.reshape(-1)
    return np.stack([sources, targets]), dists


def _radius_edges(coords: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    nbrs = NearestNeighbors(radius=radius, metric="euclidean").fit(coords)
    distances, indices = nbrs.radius_neighbors(coords, sort_results=True)
    src_list, tgt_list, dist_list = [], [], []
    for i, (dist_i, idx_i) in enumerate(zip(distances, indices)):
        mask = idx_i != i
        src_list.append(np.full(np.sum(mask), i))
        tgt_list.append(idx_i[mask])
        dist_list.append(dist_i[mask])
    if len(src_list) == 0 or np.sum([len(x) for x in src_list]) == 0:
        raise ValueError("Radius graph produced no edges; consider increasing radius.")
    sources = np.concatenate(src_list)
    targets = np.concatenate(tgt_list)
    dists = np.concatenate(dist_list)
    return np.stack([sources, targets]), dists


def build_spatial_graph(
    adata: "sc.AnnData",
    k: int = 8,
    radius: Optional[float] = None,
    rbf_dim: int = 16,
    rbf_min: Optional[float] = None,
    rbf_max: Optional[float] = None,
) -> GraphArtifacts:
    """
    Construct PyG Data object with spatial edges and distance encodings.
    """
    if "spatial" not in adata.obsm:
        raise ValueError("AnnData is missing obsm['spatial'] coordinates.")
    coords = np.array(adata.obsm["spatial"])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Spatial coordinates must have shape (n_spots, 2).")

    if radius is not None:
        edge_index_np, dists = _radius_edges(coords, radius)
        logger.info("Built radius graph with radius=%.3f and %d edges", radius, edge_index_np.shape[1])
    else:
        edge_index_np, dists = _knn_edges(coords, k)
        logger.info("Built kNN graph with k=%d and %d edges", k, edge_index_np.shape[1])

    rbf, centers = radial_basis_encoding(dists, num_centers=rbf_dim, d_min=rbf_min, d_max=rbf_max)

    x = adata.X
    if not isinstance(x, np.ndarray):
        x = x.A  # convert sparse to dense

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index_np, dtype=torch.long),
        edge_attr=torch.tensor(rbf, dtype=torch.float32),
        pos=torch.tensor(coords, dtype=torch.float32),
    )
    data.obs_names = adata.obs_names.to_list()
    data.distances = torch.tensor(dists, dtype=torch.float32)
    data.rbf_centers = torch.tensor(centers, dtype=torch.float32)
    return GraphArtifacts(data=data, distances=dists, rbf_centers=centers)


def save_graph(graph: Data, path: Path) -> None:
    """Persist PyG Data object."""
    ensure_dir(path.parent)
    torch.save(graph, path)
