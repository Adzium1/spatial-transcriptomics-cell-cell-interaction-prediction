"""End-to-end pipeline from Visium outs to figures and tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import scanpy as sc
import pandas as pd
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from spatial_interactions.graph.build_graph import build_spatial_graph, save_graph  # noqa: E402
from spatial_interactions.models.edge_head import EdgeScoreHead  # noqa: E402
from spatial_interactions.models.gatv2_edge import DistanceAwareGATv2  # noqa: E402
from spatial_interactions.preprocessing.features import select_hvgs  # noqa: E402
from spatial_interactions.preprocessing.qc_normalize import filter_genes_by_pct, filter_spots, normalize_log1p  # noqa: E402
from spatial_interactions.preprocessing.visium_loader import load_visium  # noqa: E402
from spatial_interactions.training.train_ssl import SSLTrainingConfig, train_ssl  # noqa: E402
from spatial_interactions.utils.io import ensure_dir, load_yaml  # noqa: E402
from spatial_interactions.utils.logging import get_logger  # noqa: E402
from spatial_interactions.visualization.embedding_plots import plot_embeddings_umap  # noqa: E402
from spatial_interactions.visualization.lr_analysis import rank_ligand_receptor_pairs  # noqa: E402
from spatial_interactions.visualization.spatial_plots import plot_spatial_edges  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full spatial interaction pipeline.")
    parser.add_argument("--visium_path", type=Path, required=True, help="Path to Space Ranger outs/")
    parser.add_argument("--run_name", type=str, default="demo", help="Name for results folder")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--lr_pairs", type=Path, default=REPO_ROOT / "data" / "lr_db" / "lr_pairs.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    results_dir = REPO_ROOT / "results" / args.run_name
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    checkpoints_dir = results_dir / "checkpoints"
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)
    ensure_dir(checkpoints_dir)

    # 1. Load and preprocess
    adata = load_visium(args.visium_path)
    adata = filter_spots(adata)
    adata = filter_genes_by_pct(adata, min_pct=cfg["preprocessing"]["min_pct"])
    adata = normalize_log1p(adata)
    adata = select_hvgs(
        adata,
        n_top_genes=cfg["preprocessing"]["n_hvg"],
        flavor=cfg["preprocessing"].get("flavor", "seurat_v3"),
    )
    processed_path = REPO_ROOT / "data" / "processed" / f"{args.run_name}.h5ad"
    ensure_dir(processed_path.parent)
    adata.write(processed_path)

    # 2. Graph
    artifacts = build_spatial_graph(
        adata,
        k=cfg["graph"]["k"],
        rbf_dim=cfg["graph"]["rbf_dim"],
    )
    graph_path = REPO_ROOT / "data" / "processed" / f"{args.run_name}_graph.pt"
    save_graph(artifacts.data, graph_path)

    # 3. Train
    train_cfg = SSLTrainingConfig(
        hidden_dim=cfg["training"]["hidden_dim"],
        out_dim=cfg["training"]["out_dim"],
        num_layers=cfg["training"]["num_layers"],
        heads=cfg["training"]["heads"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        epochs=cfg["training"]["epochs"],
        val_frac=cfg["training"]["val_frac"],
        neg_ratio=cfg["training"]["neg_ratio"],
        patience=cfg["training"]["patience"],
        grad_clip=cfg["training"]["grad_clip"],
        seed=cfg.get("seed", 42),
        use_amp=cfg["training"].get("use_amp", False),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    metrics = train_ssl(artifacts.data, train_cfg, results_dir)
    logger.info("Training metrics: %s", metrics)

    # 4. Load best model to score edges
    ckpt = torch.load(results_dir / "checkpoints" / "best_model.pt", map_location="cpu")["model_state"]
    graph = torch.load(graph_path)
    encoder = DistanceAwareGATv2(
        in_dim=graph.x.size(1),
        hidden_dim=train_cfg.hidden_dim,
        out_dim=train_cfg.out_dim,
        num_layers=train_cfg.num_layers,
        heads=train_cfg.heads,
        edge_dim=graph.edge_attr.size(1),
    )
    head = EdgeScoreHead(in_dim=train_cfg.out_dim, edge_dim=graph.edge_attr.size(1))
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])
    encoder.eval()
    head.eval()
    with torch.no_grad():
        z = encoder(graph.x, graph.edge_index, graph.edge_attr)
        logits = head(z, graph.edge_index, graph.edge_attr)
        scores = torch.sigmoid(logits).cpu().numpy()
    embeddings = z.cpu().numpy()

    # 5. Figures
    coords = adata.obsm["spatial"]
    plot_spatial_edges(coords, graph.edge_index.cpu().numpy(), scores, figures_dir / "spatial_interaction_edges.png", top_k=cfg["visualization"]["top_k_edges"])
    plot_embeddings_umap(embeddings, adata.obs, figures_dir / "umap_embeddings.png")

    # 6. Tables
    df_edges = pd.DataFrame(
        {
            "source_barcode": [graph.obs_names[s] if hasattr(graph, "obs_names") else s for s in graph.edge_index[0].numpy()],
            "target_barcode": [graph.obs_names[t] if hasattr(graph, "obs_names") else t for t in graph.edge_index[1].numpy()],
            "score": scores,
            "dist": graph.distances.cpu().numpy() if hasattr(graph, "distances") else float("nan"),
            "source_x": coords[graph.edge_index[0].numpy(), 0],
            "source_y": coords[graph.edge_index[0].numpy(), 1],
            "target_x": coords[graph.edge_index[1].numpy(), 0],
            "target_y": coords[graph.edge_index[1].numpy(), 1],
        }
    ).sort_values("score", ascending=False)
    df_edges.to_csv(tables_dir / "top_edges.csv", index=False)

    lr_df = rank_ligand_receptor_pairs(adata, graph.edge_index.cpu().numpy(), scores, args.lr_pairs)
    if lr_df is not None:
        lr_df.to_csv(tables_dir / "top_lr_pairs.csv", index=False)

    logger.info("End-to-end run complete. Results stored at %s", results_dir)


if __name__ == "__main__":
    main()
