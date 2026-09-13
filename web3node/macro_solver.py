"""Lean macro/energy shift probe. Not Gemini's parquet vault."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any
import numpy as np
from fieldcore_bridge import pick_juniorllm_port, stocksnode_manifold
from sparse_formats import compare_formats
from svd_residual_tick import residual_tick
from ternary_tda import TernaryTDAMesh, bitnet_quantize
from vietoris_rips import vr_summary

def gemini_threshold(x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    scale = np.mean(np.abs(x), axis=-1, keepdims=True) + 1e-7
    scaled = x / scale
    return np.where(scaled > threshold, 1.0, np.where(scaled < -threshold, -1.0, 0.0))

def synthetic_ts(n: int = 48, d: int = 16, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    energy = np.cumsum(rng.normal(scale=0.4, size=(n, d)), axis=0)
    macro = np.cumsum(rng.normal(scale=0.6, size=(n, d)), axis=0)
    return np.concatenate([energy, macro], axis=1)

def as_returns(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        return x
    prev = np.maximum(np.abs(x[:, :-1]), 1e-8)
    return np.diff(x, axis=1) / prev

def load_matrix(prefer_stocks: bool = True) -> np.ndarray:
    if prefer_stocks:
        try:
            close = np.asarray(stocksnode_manifold(n=32, t=24, seed=11)["close"])
            return as_returns(close)
        except Exception:
            pass
    return as_returns(synthetic_ts())

def solve(matrix: np.ndarray | None = None) -> dict[str, Any]:
    raw = load_matrix() if matrix is None else np.asarray(matrix, dtype=np.float64)
    x = raw if float(np.mean(np.abs(raw))) < 2.0 else as_returns(raw)
    q_abs, gamma = bitnet_quantize(x)
    q_abs = np.asarray(q_abs)
    q_thr = gemini_threshold(x)
    sample = q_abs[: min(24, q_abs.shape[0])]
    tick = residual_tick(x, mesh=TernaryTDAMesh(drift_threshold=0.12), energy=0.90)
    row = {
        "t": int(time.time() * 1000),
        "shape": list(x.shape),
        "port": pick_juniorllm_port("field energy macro"),
        "gamma": float(gamma),
        "absmean_sparsity": float((q_abs == 0).mean()),
        "threshold_sparsity": float((q_thr == 0).mean()),
        "absmean_vs_threshold_agree": float((q_abs == q_thr).mean()),
        "vr": vr_summary(sample),
        "tick": {"k": tick.get("k"), "retained_energy": tick.get("retained_energy"), "drift": tick.get("drift")},
        "sparse": compare_formats(x),
        "gemini_rejected": ["parquet required ingest", "02_Assets vault", "SVD is dead", "var-as-Betti"],
    }
    out = Path(__file__).resolve().parent / "macro_solver.jsonl"
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
    row["jsonl"] = str(out)
    return row

if __name__ == "__main__":
    print(json.dumps(solve(), indent=2, default=str))
