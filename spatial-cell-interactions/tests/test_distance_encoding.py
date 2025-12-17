from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from spatial_interactions.graph.build_graph import radial_basis_encoding


def test_rbf_encoding_shape_and_centers():
    distances = np.array([0.1, 0.5, 1.0], dtype=float)
    enc, centers = radial_basis_encoding(distances, num_centers=5)
    assert enc.shape == (len(distances), 5)
    assert np.all(np.diff(centers) > 0)
    # ensure encodings are finite and within [0,1]
    assert np.all(np.isfinite(enc))
    assert enc.max() <= 1.0 + 1e-6
