"""
Lightweight smoke test: load h5ad + graph and run a 1-epoch SSL train on CPU.
Skips gracefully if demo data/graph are missing (keeps CI green without large artifacts).
"""

from __future__ import annotations

from pathlib import Path

import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from spatial_interactions.training.train_ssl import SSLTrainingConfig, train_ssl  # noqa: E402
from spatial_interactions.utils.io import ensure_dir  # noqa: E402


def main() -> None:
    h5ad = Path("data/processed/breast_cytassist_ffpe.h5ad")
    graph_path = Path("data/processed/breast_cytassist_ffpe_radius_graph.pt")
    if not (h5ad.exists() and graph_path.exists()):
        print("Smoke test skipped: demo data/graph not present.")
        return

    # Allow PyG types in torch.load
    from torch.serialization import add_safe_globals
    from torch_geometric.data import Data
    try:
        from torch_geometric.data import DataEdgeAttr  # PyG >=2.5
        add_safe_globals([Data, DataEdgeAttr])
    except ImportError:
        add_safe_globals([Data])
    graph = torch.load(graph_path, weights_only=False, map_location="cpu")

    cfg = SSLTrainingConfig(
        hidden_dim=32,
        out_dim=16,
        num_layers=2,
        heads=2,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=1,
        val_frac=0.1,
        neg_ratio=1.0,
        patience=1,
        grad_clip=1.0,
        seed=42,
        use_amp=False,
        device="cpu",
    )
    out_dir = REPO_ROOT / "results" / "smoke_test"
    ensure_dir(out_dir)
    train_ssl(graph, cfg, out_dir)
    print("Smoke test passed: training ran for 1 epoch.")


if __name__ == "__main__":
    main()
