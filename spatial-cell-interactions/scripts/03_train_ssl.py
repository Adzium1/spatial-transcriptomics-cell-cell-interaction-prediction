"""Train self-supervised distance-aware GATv2."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from spatial_interactions.training.train_ssl import SSLTrainingConfig, train_ssl  # noqa: E402
from spatial_interactions.utils.io import ensure_dir, load_yaml, save_yaml  # noqa: E402
from spatial_interactions.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train self-supervised GATv2 on spatial graph.")
    parser.add_argument("--graph", type=Path, required=True, help="Graph .pt file")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory for run artifacts")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu or cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_yaml = load_yaml(args.config)
    train_cfg = SSLTrainingConfig(
        hidden_dim=cfg_yaml["training"]["hidden_dim"],
        out_dim=cfg_yaml["training"]["out_dim"],
        num_layers=cfg_yaml["training"]["num_layers"],
        heads=cfg_yaml["training"]["heads"],
        lr=cfg_yaml["training"]["lr"],
        weight_decay=cfg_yaml["training"]["weight_decay"],
        epochs=cfg_yaml["training"]["epochs"],
        val_frac=cfg_yaml["training"]["val_frac"],
        neg_ratio=cfg_yaml["training"]["neg_ratio"],
        patience=cfg_yaml["training"]["patience"],
        grad_clip=cfg_yaml["training"]["grad_clip"],
        seed=cfg_yaml.get("seed", 42),
        use_amp=cfg_yaml["training"].get("use_amp", False),
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = REPO_ROOT / "results" / f"run_{timestamp}"
    ensure_dir(args.out_dir)
    save_yaml(cfg_yaml, args.out_dir / "config_used.yaml")

    graph = torch.load(args.graph)
    metrics = train_ssl(graph, train_cfg, args.out_dir)
    logger.info("Training complete. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
