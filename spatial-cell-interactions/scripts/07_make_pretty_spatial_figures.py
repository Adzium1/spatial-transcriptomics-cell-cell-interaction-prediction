"""Generate tightly cropped spatial figures with histology alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from torch.serialization import add_safe_globals

# ensure package import works when run as script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from spatial_interactions.utils.io import ensure_dir
from spatial_interactions.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create publication-ready spatial figures.")
    p.add_argument("--h5ad", type=Path, required=True, help="Processed AnnData file")
    p.add_argument("--graph", type=Path, required=False, help="Graph .pt file for overlay")
    p.add_argument("--padding", type=float, default=200.0, help="Padding in pixels for cropping")
    p.add_argument(
        "--img_key",
        type=str,
        default=None,
        help="Image key to use (hires, lowres, or none). Defaults to hires if available.",
    )
    p.add_argument("--top_edges", type=int, default=2000, help="Number of edges to overlay if graph provided")
    p.add_argument(
        "--crop_mode",
        choices=["auto", "spots", "image"],
        default="auto",
        help="auto: let scanpy crop; spots: crop to spots (in_tissue); image: crop to tissue region in image via intensity mask",
    )
    return p.parse_args()


def _crop_from_coords(coords: np.ndarray, padding: float) -> tuple[float, float, float, float]:
    left = float(np.min(coords[:, 0]) - padding)
    right = float(np.max(coords[:, 0]) + padding)
    top = float(np.min(coords[:, 1]) - padding)
    bottom = float(np.max(coords[:, 1]) + padding)
    return left, right, top, bottom


def _crop_from_image(image: np.ndarray, padding: float) -> tuple[float, float, float, float] | None:
    """Compute crop from tissue image by thresholding background."""
    img = image.astype(float)
    if img.ndim == 3:
        gray = img.mean(axis=2)
    else:
        gray = img
    thresh = gray.max() * 0.98
    mask = gray < thresh
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask)
    left = float(xs.min() - padding)
    right = float(xs.max() + padding)
    top = float(ys.min() - padding)
    bottom = float(ys.max() + padding)
    return left, right, top, bottom


def _spot_size_from_scalefactors(adata: "sc.AnnData", img_key: str) -> float:
    try:
        lib = list(adata.uns["spatial"].keys())[0]
        sf = adata.uns["spatial"][lib]["scalefactors"]
        diameter_px = sf.get("spot_diameter_fullres", None)
        if diameter_px is None:
            return 1.0
        if img_key == "hires":
            return float(diameter_px)
        scale = sf.get(f"tissue_{img_key}_scalef", None)
        if scale is None:
            return float(diameter_px)
        return float(diameter_px * scale)
    except Exception:
        return 1.0


def main() -> None:
    args = parse_args()
    adata = sc.read_h5ad(args.h5ad)
    lib = list(adata.uns["spatial"].keys())[0]
    if args.img_key and args.img_key.lower() == "none":
        img_key = None
    else:
        img_key = args.img_key or ("hires" if "hires" in adata.uns["spatial"][lib]["images"] else "lowres")

    # Ensure QC metrics for counts plot
    if "total_counts" not in adata.obs.columns:
        sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None)

    fig_dir = Path("results") / "figures"
    ensure_dir(fig_dir)

    coords = adata.obsm["spatial"]
    mask = adata.obs["in_tissue"] == 1 if "in_tissue" in adata.obs else np.ones(adata.n_obs, dtype=bool)
    crop = None
    if args.crop_mode == "spots":
        crop = _crop_from_coords(coords[mask], args.padding)
    elif args.crop_mode == "image":
        img = adata.uns["spatial"][lib]["images"][img_key]
        crop = _crop_from_image(img, args.padding)
        if crop is None:
            logger.warning("Image-based crop failed; falling back to spot-based crop.")
            crop = _crop_from_coords(coords[mask], args.padding)
    spot_size = _spot_size_from_scalefactors(adata, img_key) if img_key else 1.0

    # Plot in_tissue categorical
    plt.figure(figsize=(6, 6))
    sc.pl.spatial(
        adata,
        library_id=lib,
        img_key=img_key,
        color="in_tissue",
        size=1.2,
        alpha_img=1.0 if img_key else 0.0,
        crop_coord=crop,
        spot_size=spot_size,
        show=False,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_dir / "spatial_in_tissue_hires.png", dpi=250, bbox_inches="tight")
    plt.close()

    # Plot total counts
    plt.figure(figsize=(6, 6))
    sc.pl.spatial(
        adata,
        library_id=lib,
        img_key=img_key,
        color="total_counts",
        size=1.2,
        alpha_img=1.0 if img_key else 0.0,
        crop_coord=crop,
        spot_size=spot_size,
        show=False,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_dir / "spatial_total_counts_hires.png", dpi=250, bbox_inches="tight")
    plt.close()

    # Edge overlay if graph provided
    if args.graph:
        try:
            from torch_geometric.data import Data  # noqa: F401
            add_safe_globals([Data])
        except Exception:
            pass
        graph = torch.load(args.graph, map_location="cpu", weights_only=False)
        edge_index = graph.edge_index.numpy()
        rng = np.random.default_rng(0)
        keep = rng.choice(edge_index.shape[1], size=min(args.top_edges, edge_index.shape[1]), replace=False)
        ei = edge_index[:, keep]

        x, y = coords[:, 0], coords[:, 1]
        plt.figure(figsize=(7, 7))
        sc.pl.spatial(
            adata,
            library_id=lib,
            img_key=img_key,
            color=None,
            size=1.0,
            alpha_img=1.0 if img_key else 0.0,
            crop_coord=crop,
            spot_size=spot_size,
            show=False,
        )
        ax = plt.gca()
        for a, b in ei.T:
            ax.plot([x[a], x[b]], [y[a], y[b]], color="tab:red", alpha=0.1, linewidth=0.6)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fig_dir / "spatial_radius_graph_overlay_hires.png", dpi=250, bbox_inches="tight")
        plt.close()

    logger.info("Saved figures to %s", fig_dir)


if __name__ == "__main__":
    sys.exit(main())
