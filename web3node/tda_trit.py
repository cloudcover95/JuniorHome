"""Modern TDA. Lean VR + H0. No extra package required."""
from __future__ import annotations
from typing import Any
import numpy as np
from ph_algorithms import h0_persistence
from vietoris_rips import vr_summary

def compute_persistent_homology(tensor_state: Any, max_homology_dim: int = 1):
    x = np.asarray(tensor_state, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[0] > 48:
        x = x[:48]
    if x.shape[1] > 8:
        x = x[:, :8]
    sk = vr_summary(x)
    return {"backend": "lean-vr", "edges": sk["edges"], "beta0": sk["betti0"],
            "h1_graph": sk["h1_graph"], "h0": h0_persistence(x),
            "max_homology_dim": max_homology_dim, "optional_extra": False}
