"""Lean VR → Home / Omega / FrameForge. UE off on T0."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from home_kernel import dispatch
from ph_algorithms import h0_persistence
from vr_opt import pairwise_sq, vr_edges_opt
from vietoris_rips import vr_summary

def cloud(n=24, d=3, seed=3):
    return np.random.default_rng(seed).normal(size=(n, d))

def run(points=None):
    pts = np.asarray(points if points is not None else cloud(), dtype=np.float64)
    d2 = pairwise_sq(pts)
    eps = float(np.sqrt(np.median(d2[np.triu_indices(pts.shape[0], 1)]))) if pts.shape[0] > 1 else 0.0
    edges = vr_edges_opt(pts, eps)
    classic = vr_summary(pts, eps)
    h0 = h0_persistence(pts)
    kernel = dispatch("frameforge vr skeleton", domain="game", watts=45)
    vault = Path(__file__).resolve().parent / "vault" / "vr"
    vault.mkdir(parents=True, exist_ok=True)
    omega = {"consumer": "JuniorOmega", "kind": "vr-skeleton", "status": "staged",
             "points": pts.tolist(), "edges": edges.tolist()}
    (vault / "omega_vr.json").write_text(json.dumps(omega), encoding="utf-8")
    ff = {"consumer": "FrameForge", "ue5": False, "surface": kernel["surface"],
          "vertices": pts[:, :3].tolist() if pts.shape[1] >= 3 else pts.tolist(),
          "edges": edges.tolist()}
    (vault / "frameforge_vr.json").write_text(json.dumps(ff), encoding="utf-8")
    return {"algo": ["pairwise_sq", "vr_1skeleton", "h0_union_find", "graph_H1"],
            "n": int(pts.shape[0]), "eps": eps, "edges": int(edges.shape[0]),
            "beta0": classic["betti0"], "h1_graph": classic["h1_graph"],
            "h0_mean_death": h0.get("mean_death"), "ue_boot": kernel["ue_boot"],
            "surface": kernel["surface"], "omega": str(vault / "omega_vr.json"),
            "frameforge": str(vault / "frameforge_vr.json")}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
