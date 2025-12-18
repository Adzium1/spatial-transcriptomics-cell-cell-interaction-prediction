"""Train supervised edge classifiers for ligand–receptor and cell-type interaction proxies."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.serialization import add_safe_globals


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_graph(graph_path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    try:
        from torch_geometric.data import Data, DataEdgeAttr  # type: ignore

        add_safe_globals([Data, DataEdgeAttr])
    except Exception:
        try:
            from torch_geometric.data import Data  # type: ignore

            add_safe_globals([Data])
        except Exception:
            pass
    g = torch.load(graph_path, weights_only=False, map_location="cpu")
    edge_index = g.edge_index.cpu()
    pos = g.pos.cpu() if hasattr(g, "pos") else None
    edge_attr = g.edge_attr.cpu() if hasattr(g, "edge_attr") else None
    return edge_index, pos, edge_attr


def _load_embeddings(args, adata, fallback_dim: int = 50) -> np.ndarray:
    if args.use_ssl_embeddings:
        node_emb_path = Path(args.ssl_run_dir) / "checkpoints" / "node_embeddings.pt"
        if node_emb_path.exists():
            try:
                obj = torch.load(node_emb_path, map_location="cpu")
                if isinstance(obj, dict) and "embeddings" in obj:
                    return obj["embeddings"].cpu().numpy().astype(np.float32)
                if torch.is_tensor(obj):
                    return obj.cpu().numpy().astype(np.float32)
            except Exception:
                pass
    # Fallback: PCA on log1p expression
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = np.log1p(X.astype(np.float32))
    pca = PCA(n_components=min(fallback_dim, X.shape[1], X.shape[0]-1))
    Z = pca.fit_transform(X)
    return Z.astype(np.float32)


def _load_labels(path: Path, csv_fallback: bool = True) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            if csv_fallback and path.with_suffix(".csv").exists():
                return pd.read_csv(path.with_suffix(".csv"))
    elif path.with_suffix(".csv").exists() and csv_fallback:
        return pd.read_csv(path.with_suffix(".csv"))
    raise FileNotFoundError(f"Labels file not found: {path} (or .csv fallback)")


def _edge_features(
    edge_ids: np.ndarray,
    z: np.ndarray,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    pos: Optional[torch.Tensor],
) -> np.ndarray:
    u = edge_index[0, edge_ids].numpy()
    v = edge_index[1, edge_ids].numpy()
    zu = z[u]
    zv = z[v]
    feats = [zu, zv, np.abs(zu - zv), zu * zv]
    if edge_attr is not None:
        ea = edge_attr[edge_ids].numpy()
        feats.append(ea)
    elif pos is not None:
        coords = pos.numpy()
        dist = np.linalg.norm(coords[u] - coords[v], axis=1, keepdims=True)
        feats.append(dist.astype(np.float32))
    return np.concatenate(feats, axis=1).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _train_task(
    feats: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    task: str,
    hidden_dim: int,
    lr: float,
    epochs: int,
    patience: int,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    x = torch.tensor(feats, dtype=torch.float32, device=device)
    y = torch.tensor(labels, device=device)
    mask_train = torch.tensor(splits == "train", device=device)
    mask_val = torch.tensor(splits == "val", device=device)
    mask_test = torch.tensor(splits == "test", device=device)

    n_classes = int(labels.max()) + 1 if labels.ndim == 1 else labels.shape[1]
    binary = n_classes == 2
    model = MLP(feats.shape[1], hidden_dim, 1 if binary else n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_metric = -np.inf
    best_state = None
    wait = 0

    def eval_split(pred_logits, y_true, mask, is_binary):
        if mask.sum().item() == 0:
            return {"auroc": np.nan, "ap": np.nan, "f1": np.nan}
        y_true_np = y_true[mask].cpu().numpy()
        if is_binary:
            probs = torch.sigmoid(pred_logits[mask]).detach().cpu().numpy().ravel()
            try:
                auroc = roc_auc_score(y_true_np, probs)
            except Exception:
                auroc = np.nan
            try:
                ap = average_precision_score(y_true_np, probs)
            except Exception:
                ap = np.nan
            return {"auroc": auroc, "ap": ap}
        else:
            probs = torch.softmax(pred_logits[mask], dim=1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            f1 = f1_score(y_true_np, preds, average="macro")
            return {"f1": f1}

    for epoch in range(epochs):
        model.train()
        logits = model(x)
        if binary:
            loss = nn.functional.binary_cross_entropy_with_logits(logits[mask_train].view(-1), y.float()[mask_train])
        else:
            loss = nn.functional.cross_entropy(logits[mask_train], y.long()[mask_train])
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits_eval = model(x)
        val_metrics = eval_split(logits_eval, y, mask_val, binary)
        metric_val = val_metrics.get("auroc", val_metrics.get("f1", 0.0))
        if metric_val > best_metric:
            best_metric = metric_val
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_all = model(x)
    if binary:
        probs = torch.sigmoid(logits_all).cpu().numpy().ravel()
        preds = (probs >= 0.5).astype(int)
    else:
        probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()
        preds = probs_all.argmax(axis=1)
        probs = probs_all.max(axis=1)

    # Final metrics on val/test
    final_metrics = {}
    if binary:
        for split_name, mask in [("val", mask_val), ("test", mask_test)]:
            if mask.sum() > 0:
                try:
                    auroc = roc_auc_score(y[mask].cpu().numpy(), probs[mask.cpu().numpy()])
                except Exception:
                    auroc = np.nan
                try:
                    ap = average_precision_score(y[mask].cpu().numpy(), probs[mask.cpu().numpy()])
                except Exception:
                    ap = np.nan
                final_metrics[f"{task}_{split_name}_auroc"] = float(auroc)
                final_metrics[f"{task}_{split_name}_ap"] = float(ap)
    else:
        for split_name, mask in [("val", mask_val), ("test", mask_test)]:
            if mask.sum() > 0:
                f1 = f1_score(y[mask].cpu().numpy(), preds[mask.cpu().numpy()], average="macro")
                final_metrics[f"{task}_{split_name}_f1"] = float(f1)

    return preds, final_metrics, probs


def main():
    ap = argparse.ArgumentParser(description="Supervised edge classifiers for LR and cell-type tasks.")
    ap.add_argument("--h5ad", type=Path, required=True)
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--labels_dir", type=Path, default=Path("data/processed/supervised_labels"))
    ap.add_argument("--use_ssl_embeddings", type=int, default=1)
    ap.add_argument("--ssl_run_dir", type=Path, default=Path("results/run_breast_ssl"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/run_breast_supervised"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    _set_seed(args.seed)
    device = torch.device(args.device)

    adata = sc.read_h5ad(args.h5ad)
    edge_index, pos, edge_attr = _load_graph(args.graph)
    z = _load_embeddings(args, adata)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, float] = {}

    def run_binary(task_name: str, df_path: Path, y_col: str, out_pred: Path):
        nonlocal metrics
        if not df_path.exists():
            print(f"Skipping {task_name}: missing {df_path}")
            return
        df = _load_labels(df_path)
        edge_ids = df["edge_id"].to_numpy()
        feats = _edge_features(edge_ids, z, edge_index, edge_attr, pos)
        labels = df[y_col].to_numpy().astype(int)
        splits = df["split"].to_numpy()
        preds, m, probs = _train_task(
            feats=feats,
            labels=labels,
            splits=splits,
            task=task_name,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            device=device,
        )
        metrics.update(m)
        out = df.copy()
        out["y_true"] = labels
        out["y_pred"] = preds
        out["y_prob"] = probs
        out.to_parquet(out_pred, index=False)
        print(f"Saved predictions for {task_name} to {out_pred}")

    # LR binary
    run_binary("lr", args.labels_dir / "lr_edge_labels.parquet", "y_lr", out_dir / "preds_lr.parquet")

    # Immune-epithelial binary
    run_binary(
        "immune_epi",
        args.labels_dir / "celltype_edge_labels_binary.parquet",
        "y_immune_epi",
        out_dir / "preds_immune_epi.parquet",
    )

    # Multiclass type_pair
    pair_path = args.labels_dir / "celltype_edge_labels_multiclass.parquet"
    if pair_path.exists():
        df = _load_labels(pair_path)
        classes = sorted(df["type_pair"].unique())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        df["y_idx"] = df["type_pair"].map(class_to_idx)
        edge_ids = df["edge_id"].to_numpy()
        feats = _edge_features(edge_ids, z, edge_index, edge_attr, pos)
        labels = df["y_idx"].to_numpy().astype(int)
        splits = df["split"].to_numpy()
        preds_idx, m, probs = _train_task(
            feats=feats,
            labels=labels,
            splits=splits,
            task="type_pair",
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            device=device,
        )
        metrics.update(m)
        out = df.copy()
        out["y_true"] = df["y_idx"]
        out["y_pred"] = preds_idx
        out["y_prob"] = probs
        out.to_parquet(out_dir / "preds_type_pair.parquet", index=False)
        with open(out_dir / "classes_type_pair.json", "w") as fh:
            json.dump(class_to_idx, fh, indent=2)
        print(f"Saved predictions for type_pair to {out_dir / 'preds_type_pair.parquet'}")
    else:
        print("Skipping type_pair: labels not found.")

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
