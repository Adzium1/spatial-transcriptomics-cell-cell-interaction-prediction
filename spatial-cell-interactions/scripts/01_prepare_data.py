"""Load Visium data, filter, normalize, and select HVGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import scanpy as sc

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from spatial_interactions.preprocessing.visium_loader import load_visium  # noqa: E402
from spatial_interactions.preprocessing.qc_normalize import filter_genes_by_pct, filter_spots, normalize_log1p  # noqa: E402
from spatial_interactions.preprocessing.features import select_hvgs  # noqa: E402
from spatial_interactions.utils.io import ensure_dir, load_yaml  # noqa: E402
from spatial_interactions.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Visium data (QC, normalize, HVGs).")
    parser.add_argument("--visium_path", type=Path, required=True, help="Path to Space Ranger outs/ directory")
    parser.add_argument("--out_h5ad", type=Path, default=None, help="Output .h5ad path")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--min_pct", type=float, default=None, help="Minimum percent of spots for gene retention")
    parser.add_argument("--n_hvg", type=int, default=None, help="Number of highly variable genes to keep")
    parser.add_argument("--count_file", type=str, default="filtered_feature_bc_matrix.h5")
    parser.add_argument("--source_image_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    min_pct = args.min_pct if args.min_pct is not None else cfg["preprocessing"]["min_pct"]
    n_hvg = args.n_hvg if args.n_hvg is not None else cfg["preprocessing"]["n_hvg"]
    flavor = cfg["preprocessing"].get("flavor", "seurat_v3")

    adata = load_visium(args.visium_path, count_file=args.count_file, source_image_path=args.source_image_path)
    adata = filter_spots(adata)
    adata = filter_genes_by_pct(adata, min_pct=min_pct)
    adata = normalize_log1p(adata)
    adata = select_hvgs(adata, n_top_genes=n_hvg, flavor=flavor, subset=True)

    if args.out_h5ad is None:
        sample = args.visium_path.resolve().parent.name
        args.out_h5ad = REPO_ROOT / "data" / "processed" / f"{sample}.h5ad"
    ensure_dir(args.out_h5ad.parent)
    logger.info("Saving processed AnnData to %s", args.out_h5ad)
    adata.write(args.out_h5ad)


if __name__ == "__main__":
    main()
