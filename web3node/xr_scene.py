"""Junior XR scene. WebXR/three.js/XR Blocks optional. No Gemini."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from home_kernel import dispatch
from ternary_kernel import kernel_list
from vr_opt import pairwise_sq, vr_edges_opt

def scene(n=16):
    pts = np.random.default_rng(4).normal(size=(n, 3))
    d2 = pairwise_sq(pts)
    eps = float(np.sqrt(np.median(d2[np.triu_indices(n, 1)])))
    edges = vr_edges_opt(pts, eps)
    ker = dispatch("omega xr webxr", domain="cad", watts=45)
    intent = kernel_list(pts[:, 0].tolist(), pts[:, 1].tolist())
    spec = {"engine": "JuniorHome", "frameworks": ["WebXR", "three.js", "xrblocks-optional"],
            "not_used": ["Gemini Canvas"], "quant": "ternary-1.58", "ue_boot": ker["ue_boot"],
            "surface": ker["surface"], "points": pts.tolist(), "edges": edges.tolist(),
            "eps": eps, "intent_y": intent["layer"]["y"],
            "xrblocks": "https://github.com/google/xrblocks"}
    vault = Path(__file__).resolve().parent / "vault" / "xr"
    vault.mkdir(parents=True, exist_ok=True)
    (vault / "scene.json").write_text(json.dumps(spec), encoding="utf-8")
    spec["out"] = str(vault / "scene.json")
    return spec

if __name__ == "__main__":
    row = scene()
    print(json.dumps({k: row[k] for k in ("engine", "frameworks", "not_used", "quant", "ue_boot", "eps", "out")}, indent=2))
