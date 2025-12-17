"""Evaluation utilities for edge reconstruction."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def edge_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute AUROC and average precision."""
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    metrics = {}
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        metrics["auroc"] = float("nan")
    try:
        metrics["ap"] = float(average_precision_score(y_true, probs))
    except ValueError:
        metrics["ap"] = float("nan")
    return metrics


def edge_predictions(
    logits: torch.Tensor, edge_index: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Return probabilities and edge indices as numpy arrays."""
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    edges = edge_index.detach().cpu().numpy()
    return probs, edges
