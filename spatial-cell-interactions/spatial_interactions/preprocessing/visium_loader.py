"""Load Visium data using Scanpy."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import scanpy as sc

from spatial_interactions.utils.logging import get_logger

logger = get_logger(__name__)


def load_visium(
    visium_path: Path,
    count_file: str = "filtered_feature_bc_matrix.h5",
    source_image_path: Optional[Path] = None,
) -> "sc.AnnData":
    """
    Load Visium Space Ranger outputs with expression, spatial coords, and images.

    Parameters
    ----------
    visium_path:
        Path to Space Ranger outs/ directory containing filtered_feature_bc_matrix.h5 and spatial/.
    count_file:
        Count matrix file name.
    source_image_path:
        Optional override for tissue image directory if not at visium_path / "spatial".
    """
    visium_path = visium_path.resolve()
    if not visium_path.exists():
        raise FileNotFoundError(
            f"Visium path {visium_path} does not exist. Expected Space Ranger outs/ directory."
        )

    count_path = visium_path / count_file
    spatial_dir = source_image_path if source_image_path is not None else visium_path / "spatial"
    if not count_path.exists():
        raise FileNotFoundError(
            f"Count file {count_path} missing. Ensure Space Ranger outputs include {count_file}."
        )
    if not spatial_dir.exists():
        raise FileNotFoundError(
            f"Spatial directory {spatial_dir} missing. Expected under Space Ranger outs/."
        )

    logger.info("Loading Visium data from %s", visium_path)
    adata = sc.read_visium(path=visium_path, count_file=count_file, load_images=True)
    if "spatial" not in adata.uns:
        raise ValueError("Scanpy did not populate adata.uns['spatial']; check input structure.")

    adata.var_names_make_unique()
    # Preserve raw counts if available
    adata.layers["counts"] = adata.X.copy()
    return adata
