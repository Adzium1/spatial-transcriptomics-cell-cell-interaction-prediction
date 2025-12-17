"""Generate figures and tables from trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from spatial_interactions.models.edge_head import EdgeScoreHead  # noqa: E402
from spatial_interactions.models.gatv2_edge import DistanceAwareGATv2  # noqa: E402
from spatial_interactions.utils.io import ensure_dir, load_yaml, save_json  # noqa: E402
from spatial_interactions.utils.logging import get_logger  # noqa: E402
from spatial_interactions.visualization.embedding_plots import plot_embeddings_umap  # noqa: E402
from spatial_interactions.visualization.lr_analysis import rank_ligand_receptor_pairs  # noqa: E402
from spatial_interactions.visualization.spatial_plots import plot_spatial_edges  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures and tables from trained model.")
    parser.add_argument("--h5ad", type=Path, required=True, help="Processed AnnData file")
    parser.add_argument("--graph", type=Path, required=True, help="Graph .pt file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint with encoder/head")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--lr_pairs", type=Path, default=REPO_ROOT / "data" / "lr_db" / "lr_pairs.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    vis_cfg = cfg.get("visualization", {})
    top_k = vis_cfg.get("top_k_edges", 500)

    adata = sc.read_h5ad(args.h5ad)
    graph = torch.load(args.graph)
    ckpt = torch.load(args.checkpoint, map_location="cpu")["model_state"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DistanceAwareGATv2(
        in_dim=graph.x.size(1),
        hidden_dim=cfg["training"]["hidden_dim"],
        out_dim=cfg["training"]["out_dim"],
        num_layers=cfg["training"]["num_layers"],
        heads=cfg["training"]["heads"],
        edge_dim=graph.edge_attr.size(1),
    ).to(device)
    head = EdgeScoreHead(in_dim=cfg["training"]["out_dim"], edge_dim=graph.edge_attr.size(1)).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])
    encoder.eval()
    head.eval()

    with torch.no_grad():
        z = encoder(graph.x.to(device), graph.edge_index.to(device), graph.edge_attr.to(device))
        logits = head(z, graph.edge_index.to(device), graph.edge_attr.to(device))
        scores = torch.sigmoid(logits).cpu().numpy()
    embeddings = z.cpu().numpy()

    if args.out_dir is None:
        args.out_dir = args.checkpoint.parent.parent if args.checkpoint.parent.parent.exists() else REPO_ROOT / "results"

    figures_dir = args.out_dir / "figures"
    tables_dir = args.out_dir / "tables"
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    coords = np.array(adata.obsm["spatial"])
    edge_index_np = graph.edge_index.cpu().numpy()
    plot_spatial_edges(coords, edge_index_np, scores, figures_dir / "spatial_interaction_edges.png", top_k=top_k)
    plot_embeddings_umap(embeddings, adata.obs, figures_dir / "umap_embeddings.png")

    # Tables
    df_edges = pd.DataFrame(
        {
            "source_barcode": [graph.obs_names[s] if hasattr(graph, "obs_names") else s for s in edge_index_np[0]],
            "target_barcode": [graph.obs_names[t] if hasattr(graph, "obs_names") else t for t in edge_index_np[1]],
            "score": scores,
            "dist": graph.distances.cpu().numpy() if hasattr(graph, "distances") else np.nan,
            "source_x": coords[edge_index_np[0], 0],
            "source_y": coords[edge_index_np[0], 1],
            "target_x": coords[edge_index_np[1], 0],
            "target_y": coords[edge_index_np[1], 1],
        }
    )
    df_edges_sorted = df_edges.sort_values("score", ascending=False)
    df_edges_sorted.to_csv(tables_dir / "top_edges.csv", index=False)

    lr_df = rank_ligand_receptor_pairs(adata, edge_index_np, scores, args.lr_pairs)
    if lr_df is not None:
        lr_df.to_csv(tables_dir / "top_lr_pairs.csv", index=False)

    meta = {
        "h5ad": str(args.h5ad),
        "graph": str(args.graph),
        "checkpoint": str(args.checkpoint),
        "top_k_edges": top_k,
    }
    save_json(meta, args.out_dir / "figures_metadata.json")
    logger.info("Figures and tables saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
