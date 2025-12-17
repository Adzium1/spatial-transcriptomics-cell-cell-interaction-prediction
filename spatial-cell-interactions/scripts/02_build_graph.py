"""Construct spatial graph with distance encodings."""

from __future__ import annotations

import argparse
from pathlib import Path

import scanpy as sc

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from spatial_interactions.graph.build_graph import build_spatial_graph, save_graph  # noqa: E402
from spatial_interactions.utils.io import ensure_dir, load_yaml  # noqa: E402
from spatial_interactions.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spatial neighbor graph from AnnData.")
    parser.add_argument("--h5ad", type=Path, required=True, help="Processed .h5ad file")
    parser.add_argument("--out_graph", type=Path, default=None, help="Output graph .pt path")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--k", type=int, default=None, help="Number of neighbors for kNN graph")
    parser.add_argument("--radius", type=float, default=None, help="Radius for radius graph (optional)")
    parser.add_argument("--rbf_dim", type=int, default=None, help="RBF embedding dimension")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    k = args.k if args.k is not None else cfg["graph"]["k"]
    rbf_dim = args.rbf_dim if args.rbf_dim is not None else cfg["graph"]["rbf_dim"]

    logger.info("Reading AnnData from %s", args.h5ad)
    adata = sc.read_h5ad(args.h5ad)
    artifacts = build_spatial_graph(adata, k=k, radius=args.radius, rbf_dim=rbf_dim)

    if args.out_graph is None:
        sample = Path(args.h5ad).stem
        args.out_graph = REPO_ROOT / "data" / "processed" / f"{sample}_graph.pt"
    ensure_dir(args.out_graph.parent)
    save_graph(artifacts.data, args.out_graph)
    logger.info("Saved graph to %s", args.out_graph)


if __name__ == "__main__":
    main()
