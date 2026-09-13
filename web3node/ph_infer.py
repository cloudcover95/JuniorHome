"""Full-stack PH inference on Home. giotto/Ripser/GUDHI named only."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from alpha_complex import summarize as alpha_summarize
from economy_energy import metrics as economy_metrics
from fieldcore_bridge import pick_juniorllm_port
from home_terraform import inject
from ph_algorithms import h0_persistence
from svd_retain import hardware_or_host_svd
from tnn_layer import bitlinear
from vietoris_rips import vr_summary
from vr_opt import bench_cloud

def svd_plane(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = hardware_or_host_svd(x)
    return x @ vt[:2].T

def infer(seed: int = 7) -> dict[str, Any]:
    econ = economy_metrics()
    rng = np.random.default_rng(seed)
    cloud = rng.normal(size=(24, 6))
    h0 = h0_persistence(cloud)
    vr = vr_summary(cloud)
    alpha = alpha_summarize(svd_plane(cloud))
    deaths = [p[1] for p in h0.get("pairs_head", [])]
    feats = deaths + [float(h0["mean_death"]), float(h0["max_death"]),
                      float(vr["edges"]), float(vr["h1_graph"]),
                      float(alpha.get("gabriel_edges", 0)), float(alpha.get("alpha_triangles", 0)),
                      float(econ["shift_ratio"]), float(econ["energy_vol"]), float(econ["macro_vol"])]
    tnn = bitlinear(feats, rng.normal(size=len(feats)).tolist())
    row = {
        "port": pick_juniorllm_port("field ph infer"),
        "terraform": {k: inject("ph infer vr alpha field").get(k) for k in ("port", "ok", "fusion_backend")},
        "h0": {k: h0[k] for k in ("n", "finite_pairs", "infinite_bars", "mean_death", "max_death")},
        "vr": vr,
        "alpha_on_svd_plane": alpha,
        "vr_opt": bench_cloud(24, 6, seed),
        "economy": {"shift_ratio": econ["shift_ratio"], "svd_k": econ["svd_k"], "trit_pack": econ["sparse"]["bytes"]["trit_pack"]},
        "tnn": {k: tnn[k] for k in ("y", "acc", "dw", "n")},
    }
    out = Path(__file__).resolve().parent / "vault" / "ph_infer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    row["out"] = str(out)
    return row

if __name__ == "__main__":
    print(json.dumps(infer(), indent=2, default=str))
