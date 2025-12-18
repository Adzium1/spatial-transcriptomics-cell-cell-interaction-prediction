"""Build proxy interaction labels (ligand-receptor and cell-type) for supervised edge modeling."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml
from sklearn.model_selection import train_test_split


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)


def _split_stratified(df: pd.DataFrame, y_col: str, seed: int) -> pd.DataFrame:
    """Add split column (train/val/test ~80/10/10)."""
    train_frac = 0.8
    val_frac = 0.1
    test_frac = 0.1
    try:
        df_train, df_tmp = train_test_split(
            df, test_size=val_frac + test_frac, random_state=seed, stratify=df[y_col]
        )
        df_val, df_test = train_test_split(
            df_tmp, test_size=test_frac / (val_frac + test_frac), random_state=seed, stratify=df_tmp[y_col]
        )
    except Exception:
        # Fallback without stratify
        df_train, df_tmp = train_test_split(df, test_size=val_frac + test_frac, random_state=seed, shuffle=True)
        df_val, df_test = train_test_split(df_tmp, test_size=test_frac / (val_frac + test_frac), random_state=seed)
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"
    return pd.concat([df_train, df_val, df_test], ignore_index=True)


def _load_graph(graph_path: Path) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    from torch.serialization import add_safe_globals

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


def _prep_expression(adata) -> Tuple[np.ndarray, Dict[str, int]]:
    var_names = adata.var_names
    upper_map = {g.upper(): i for i, g in enumerate(var_names)}
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = X.astype(np.float32)
    X_log = np.log1p(X)
    return X_log, upper_map


def _gene_expr(name: str, X: np.ndarray, gmap: Dict[str, int]) -> Optional[np.ndarray]:
    idx = gmap.get(name.upper())
    if idx is None:
        return None
    return X[:, idx]


def build_lr_labels(
    edge_index: torch.Tensor,
    X_log: np.ndarray,
    gmap: Dict[str, int],
    lr_pairs: List[Tuple[str, str]],
    pos_q: float,
    neg_q: float,
    min_expr: float,
    seed: int,
) -> pd.DataFrame:
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    scores = np.zeros(len(src), dtype=np.float32)

    # Precache expressions
    cache: Dict[str, Optional[np.ndarray]] = {}
    for l, r in lr_pairs:
        if l.upper() not in cache:
            cache[l.upper()] = _gene_expr(l, X_log, gmap)
        if r.upper() not in cache:
            cache[r.upper()] = _gene_expr(r, X_log, gmap)

    for (l, r) in lr_pairs:
        el = cache.get(l.upper())
        er = cache.get(r.upper())
        if el is None or er is None:
            continue
        # apply min_expr filter
        el_f = np.where(el >= min_expr, el, 0.0)
        er_f = np.where(er >= min_expr, er, 0.0)
        score_ij = el_f[src] * er_f[dst]
        score_ji = el_f[dst] * er_f[src]
        scores = np.maximum(scores, np.maximum(score_ij, score_ji))

    pos_thr = np.quantile(scores, pos_q) if len(scores) else 0.0
    neg_thr = np.quantile(scores, neg_q) if len(scores) else 0.0

    labels = np.full(len(scores), fill_value=-1, dtype=np.int8)
    labels[scores >= pos_thr] = 1
    labels[scores <= neg_thr] = 0
    mask = labels >= 0

    df = pd.DataFrame(
        {
            "edge_id": np.arange(len(scores))[mask],
            "src": src[mask],
            "dst": dst[mask],
            "lr_score": scores[mask],
            "y_lr": labels[mask],
        }
    )
    df = _split_stratified(df, "y_lr", seed)
    return df


def build_celltype_labels(
    adata, X_log: np.ndarray, gmap: Dict[str, int], markers: Dict[str, List[str]]
) -> pd.DataFrame:
    scores = {}
    for ct, genes in markers.items():
        vals = []
        for g in genes:
            expr = _gene_expr(g, X_log, gmap)
            if expr is not None:
                vals.append(expr)
        if len(vals) == 0:
            scores[ct] = np.zeros(X_log.shape[0], dtype=np.float32)
        else:
            arr = np.stack(vals, axis=1)
            scores[ct] = arr.mean(axis=1)
    score_mat = np.stack(list(scores.values()), axis=1)
    cts = list(scores.keys())
    max_idx = score_mat.argmax(axis=1)
    max_val = score_mat.max(axis=1)
    cell_type = np.where(max_val > 0.0, np.array(cts)[max_idx], "Unknown")
    df = pd.DataFrame({"obs_name": adata.obs_names.values, "cell_type": cell_type})
    for ct, vals in scores.items():
        df[f"score_{ct}"] = vals
    return df


def build_celltype_edge_labels(
    edge_index: torch.Tensor,
    cell_types: np.ndarray,
    immune: List[str],
    epithelial: List[str],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    ct_src = cell_types[src]
    ct_dst = cell_types[dst]

    # immune-epi binary
    immune_set = set(immune)
    epi_set = set(epithelial)
    valid = (ct_src != "Unknown") & (ct_dst != "Unknown")
    is_pos = (
        (np.isin(ct_src, list(immune_set)) & np.isin(ct_dst, list(epi_set)))
        | (np.isin(ct_dst, list(immune_set)) & np.isin(ct_src, list(epi_set)))
    )
    y_ie = np.where(valid, is_pos.astype(int), -1)
    mask_ie = y_ie >= 0
    df_ie = pd.DataFrame(
        {
            "edge_id": np.arange(len(src))[mask_ie],
            "src": src[mask_ie],
            "dst": dst[mask_ie],
            "y_immune_epi": y_ie[mask_ie],
        }
    )
    df_ie = _split_stratified(df_ie, "y_immune_epi", seed)

    # multiclass directed pair
    pair = np.where(valid, ct_src + "__" + ct_dst, "Unknown")
    df_pair = pd.DataFrame(
        {
            "edge_id": np.arange(len(src)),
            "src": src,
            "dst": dst,
            "type_pair": pair,
        }
    )
    df_pair = df_pair[df_pair["type_pair"] != "Unknown"].copy()
    counts = df_pair["type_pair"].value_counts()
    keep_classes = set(counts[counts >= 30].index)
    df_pair["type_pair"] = df_pair["type_pair"].apply(lambda x: x if x in keep_classes else "Other")
    try:
        df_pair = _split_stratified(df_pair, "type_pair", seed)
    except Exception:
        df_pair["split"] = "train"
    return df_ie, df_pair


def main():
    p = argparse.ArgumentParser(description="Build LR and cell-type interaction labels.")
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("data/processed/supervised_labels/"))
    p.add_argument("--lr_pairs", type=Path, default=Path("resources/ligand_receptor_pairs.csv"))
    p.add_argument("--markers", type=Path, default=Path("resources/marker_sets.yaml"))
    p.add_argument("--min_expr", type=float, default=0.0)
    p.add_argument("--lr_pos_quantile", type=float, default=0.90)
    p.add_argument("--lr_neg_quantile", type=float, default=0.50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    _set_seed(args.seed)

    adata = sc.read_h5ad(args.h5ad)
    edge_index, pos, edge_attr = _load_graph(args.graph)

    lr_pairs_df = pd.read_csv(args.lr_pairs)
    lr_pairs = list(zip(lr_pairs_df["ligand"], lr_pairs_df["receptor"]))
    with open(args.markers, "r") as fh:
        markers = yaml.safe_load(fh)

    X_log, gmap = _prep_expression(adata)

    lr_df = build_lr_labels(
        edge_index=edge_index,
        X_log=X_log,
        gmap=gmap,
        lr_pairs=lr_pairs,
        pos_q=args.lr_pos_quantile,
        neg_q=args.lr_neg_quantile,
        min_expr=args.min_expr,
        seed=args.seed,
    )

    celltype_df = build_celltype_labels(adata, X_log, gmap, markers)

    immune = ["T_cells", "B_cells", "Myeloid"]
    epithelial = ["Epithelial"]
    celltype_edges_bin, celltype_edges_pair = build_celltype_edge_labels(
        edge_index=edge_index,
        cell_types=celltype_df["cell_type"].values,
        immune=immune,
        epithelial=epithelial,
        seed=args.seed,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_df(lr_df, out_dir / "lr_edge_labels.parquet")
    _save_df(celltype_df, out_dir / "celltype_node_labels.parquet")
    _save_df(celltype_edges_bin, out_dir / "celltype_edge_labels_binary.parquet")
    _save_df(celltype_edges_pair, out_dir / "celltype_edge_labels_multiclass.parquet")
    print(f"Wrote labels to {out_dir}")


if __name__ == "__main__":
    main()
