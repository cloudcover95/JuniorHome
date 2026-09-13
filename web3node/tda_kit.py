"""Home TDA kit. Kruskal H0, lean VR, Flagstaff terraform."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from checks_balance import check
from home_terraform import inject
from ph_algorithms import h0_persistence
from second_brain import write_note
from vr_opt import pairwise_sq, vr_edges_opt
from vietoris_rips import vr_summary

def infer(points, label="flagstaff field tda"):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)
    d2 = pairwise_sq(pts)
    off = d2[np.triu_indices(pts.shape[0], 1)]
    eps = float(np.sqrt(np.median(off))) if off.size else 0.0
    edges = vr_edges_opt(pts, eps)
    sk = vr_summary(pts, eps)
    tf = inject(label)
    bal = check(label)
    return {"math": {"H0": "Kruskal / union-find", "stability": "CSEH 2007 bottleneck",
                     "graph_H1": "E-V+beta0"}, "n": int(pts.shape[0]), "eps": eps,
            "edges": int(edges.shape[0]), "beta0": sk["betti0"], "h1_graph": sk["h1_graph"],
            "h0": h0_persistence(pts), "terraform": {k: tf.get(k) for k in ("port", "ok")},
            "balance": {"ok": bal.get("ok"), "port": bal.get("port")},
            "flagstaff": "flagstaff" in label.lower()}

def pulse(points=None):
    pts = points if points is not None else np.random.default_rng(7).normal(size=(20, 3))
    row = infer(pts)
    row["obsidian"] = str(write_note(Path(__file__).resolve().parent / "vault",
        f"# TDA kit\n\nbeta0={row['beta0']} edges={row['edges']}\n"))
    return row

if __name__ == "__main__":
    import json
    print(json.dumps(pulse(), indent=2, default=str))
