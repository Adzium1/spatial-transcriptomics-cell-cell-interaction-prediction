"""Self-supervised edge reconstruction training loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch_geometric.data import Data

from spatial_interactions.graph.negative_sampling import sample_negative_edges
from spatial_interactions.models.edge_head import EdgeScoreHead
from spatial_interactions.models.gatv2_edge import DistanceAwareGATv2
from spatial_interactions.models.losses import bce_loss
from spatial_interactions.training.callbacks import EarlyStopping
from spatial_interactions.training.evaluate import edge_metrics
from spatial_interactions.utils.io import ensure_dir, save_json
from spatial_interactions.utils.logging import get_logger
from spatial_interactions.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class SSLTrainingConfig:
    hidden_dim: int = 64
    out_dim: int = 64
    num_layers: int = 2
    heads: int = 4
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 100
    val_frac: float = 0.1
    neg_ratio: float = 1.0
    patience: int = 15
    grad_clip: float = 5.0
    seed: int = 42
    use_amp: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _split_edges(edge_index: torch.Tensor, val_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    num_edges = edge_index.size(1)
    val_size = max(1, int(num_edges * val_frac))
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_edges, generator=generator)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return edge_index[:, train_idx], edge_index[:, val_idx], train_idx, val_idx


def _rbf_from_centers(distances: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Encode distances using pre-defined centers."""
    if centers.numel() > 1:
        width = centers[1] - centers[0]
    else:
        width = torch.tensor(1.0, device=centers.device)
    gamma = 1.0 / (2 * (width**2))
    dist = distances[:, None]
    enc = torch.exp(-gamma * (dist - centers[None, :]) ** 2)
    return enc


def _edge_attr_from_coords(
    pos: torch.Tensor, edge_index: torch.Tensor, centers: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    src, dst = edge_index
    dists = torch.norm(pos[src] - pos[dst], dim=1)
    enc = _rbf_from_centers(dists, centers)
    return dists, enc


def train_ssl(graph: Data, cfg: SSLTrainingConfig, out_dir: Path) -> Dict[str, float]:
    """Train self-supervised edge reconstruction model on a single graph."""
    set_seed(cfg.seed)
    ensure_dir(out_dir)

    device = torch.device(cfg.device)
    graph = graph.to(device)

    train_pos, val_pos, train_idx, val_idx = _split_edges(graph.edge_index, cfg.val_frac, cfg.seed)
    train_pos_attr = graph.edge_attr[train_idx]
    val_pos_attr = graph.edge_attr[val_idx]

    encoder = DistanceAwareGATv2(
        in_dim=graph.x.size(1),
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.out_dim,
        num_layers=cfg.num_layers,
        heads=cfg.heads,
        edge_dim=graph.edge_attr.size(1),
    ).to(device)
    head = EdgeScoreHead(in_dim=cfg.out_dim, edge_dim=graph.edge_attr.size(1)).to(device)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")
    stopper = EarlyStopping(patience=cfg.patience, mode="max")

    metrics_log: Dict[str, float] = {}
    for epoch in range(cfg.epochs):
        encoder.train()
        head.train()
        optimizer.zero_grad()

        neg_edges = sample_negative_edges(
            edge_index=graph.edge_index, num_nodes=graph.num_nodes, num_samples=int(train_pos.size(1) * cfg.neg_ratio)
        ).to(device)
        _, neg_attr = _edge_attr_from_coords(graph.pos, neg_edges, graph.rbf_centers.to(device))

        labels = torch.cat(
            [torch.ones(train_pos.size(1), device=device), torch.zeros(neg_edges.size(1), device=device)], dim=0
        )
        edge_batch = torch.cat([train_pos, neg_edges], dim=1)
        edge_attr_batch = torch.cat([train_pos_attr, neg_attr], dim=0)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp and device.type == "cuda"):
            z = encoder(graph.x, graph.edge_index, graph.edge_attr)
            logits = head(z, edge_batch, edge_attr_batch)
            loss = bce_loss(logits, labels)

        scaler.scale(loss).backward()
        if cfg.grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Validation
        encoder.eval()
        head.eval()
        with torch.no_grad():
            val_neg = sample_negative_edges(
                edge_index=graph.edge_index, num_nodes=graph.num_nodes, num_samples=val_pos.size(1)
            ).to(device)
            _, val_neg_attr = _edge_attr_from_coords(graph.pos, val_neg, graph.rbf_centers.to(device))
            val_labels = torch.cat(
                [torch.ones(val_pos.size(1), device=device), torch.zeros(val_neg.size(1), device=device)], dim=0
            )
            val_edges = torch.cat([val_pos, val_neg], dim=1)
            val_attr = torch.cat([val_pos_attr, val_neg_attr], dim=0)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp and device.type == "cuda"):
                z_val = encoder(graph.x, graph.edge_index, graph.edge_attr)
                val_logits = head(z_val, val_edges, val_attr)
            metrics = edge_metrics(val_logits, val_labels)
            metrics_log = {"epoch": epoch, "loss": float(loss.item()), **metrics}
        logger.info(
            "Epoch %03d | loss=%.4f | val_ap=%.4f | val_auroc=%.4f",
            epoch,
            loss.item(),
            metrics["ap"],
            metrics["auroc"],
        )

        state = {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
        }
        if stopper.step(metrics.get("ap", float("-inf")), state):
            logger.info("Early stopping at epoch %d", epoch)
            break

    # Save artifacts
    checkpoint_path = out_dir / "checkpoints" / "best_model.pt"
    stopper.save_best(checkpoint_path, encoder)
    best_state = torch.load(checkpoint_path)["model_state"]
    encoder.load_state_dict(best_state["encoder"])
    head.load_state_dict(best_state["head"])
    encoder.eval()
    with torch.no_grad():
        z_final = encoder(graph.x, graph.edge_index, graph.edge_attr).cpu()
    torch.save(
        {"embeddings": z_final, "encoder_state": encoder.state_dict(), "head_state": head.state_dict()},
        out_dir / "checkpoints" / "node_embeddings.pt",
    )
    save_json({**asdict(cfg), **metrics_log}, out_dir / "metrics.json")
    return metrics_log
