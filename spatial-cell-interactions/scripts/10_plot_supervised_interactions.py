"""Plot supervised interaction predictions (LR and immune-epithelial)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import precision_recall_curve, roc_curve


def load_preds(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)
    raise FileNotFoundError(f"Predictions not found: {path}")


def compute_crop(coords: np.ndarray, mask: np.ndarray, pad: float = 100.0):
    xy = coords[mask]
    return (
        float(xy[:, 0].min() - pad),
        float(xy[:, 0].max() + pad),
        float(xy[:, 1].min() - pad),
        float(xy[:, 1].max() + pad),
    )


def plot_edges(
    adata,
    edges: pd.DataFrame,
    title: str,
    out_path: Path,
    img_key: str,
    spots_only: bool,
    color: str,
    top_k: int,
):
    lib = list(adata.uns["spatial"].keys())[0]
    coords = adata.obsm["spatial"]
    mask = adata.obs["in_tissue"].astype(int).values == 1 if "in_tissue" in adata.obs else np.ones(adata.n_obs, dtype=bool)
    crop = compute_crop(coords, mask)

    edges_sorted = edges.sort_values("y_prob", ascending=False).head(top_k)
    x = coords[:, 0]
    y = coords[:, 1]

    plt.figure(figsize=(7, 7))
    sc.pl.spatial(
        adata,
        library_id=lib,
        img_key=None if spots_only else img_key,
        color=None,
        size=1.0,
        alpha_img=1.0 if not spots_only else 0.0,
        crop_coord=crop,
        show=False,
    )
    ax = plt.gca()
    for _, row in edges_sorted.iterrows():
        ax.plot([x[int(row.src)], x[int(row.dst)]], [y[int(row.src)], y[int(row.dst)]], color=color, alpha=0.15, linewidth=0.6)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_curves(df: pd.DataFrame, title: str, out_prefix: Path):
    y_true = df["y_true"].to_numpy()
    y_prob = df["y_prob"].to_numpy()
    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC - {title}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix.with_name(out_prefix.stem + "_roc.png"), dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        pass
    # PR
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.figure()
        plt.plot(rec, prec, label="PR")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR - {title}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix.with_name(out_prefix.stem + "_pr.png"), dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def plot_lr_scatter(df: pd.DataFrame, out_path: Path):
    if "lr_score" not in df.columns:
        return
    plt.figure(figsize=(5, 4))
    plt.scatter(df["lr_score"], df["y_prob"], s=5, alpha=0.5)
    plt.xlabel("LR score (expression proxy)")
    plt.ylabel("Predicted prob")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot supervised interaction predictions.")
    ap.add_argument("--h5ad", type=Path, required=True)
    ap.add_argument("--preds_lr", type=Path, required=True)
    ap.add_argument("--preds_immune_epi", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("results/figures_supervised"))
    ap.add_argument("--top_k", type=int, default=2000)
    ap.add_argument("--img_key", type=str, default="lowres")
    ap.add_argument("--spots_only", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    preds_lr = load_preds(args.preds_lr)
    preds_ie = load_preds(args.preds_immune_epi)

    plot_edges(
        adata=adata,
        edges=preds_lr,
        title="Top LR edges",
        out_path=out_dir / "lr_edges_topk.png",
        img_key=args.img_key,
        spots_only=bool(args.spots_only),
        color="tab:red",
        top_k=args.top_k,
    )
    plot_edges(
        adata=adata,
        edges=preds_ie,
        title="Top immune-epithelial edges",
        out_path=out_dir / "immune_epi_edges_topk.png",
        img_key=args.img_key,
        spots_only=bool(args.spots_only),
        color="tab:blue",
        top_k=args.top_k,
    )

    plot_curves(preds_lr, "LR", out_dir / "lr_curves.png")
    plot_curves(preds_ie, "Immune-Epithelial", out_dir / "immune_epi_curves.png")
    plot_lr_scatter(preds_lr, out_dir / "lr_score_vs_prob.png")
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
