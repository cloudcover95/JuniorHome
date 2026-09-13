"""Labeled C/H/L -> process_manifold. Sector split by ticker labels."""
from __future__ import annotations
from typing import Any
import numpy as np
from live_book import load_book
from ph_algorithms import h0_persistence
from sparse_formats import compare_formats
from svd_residual_tick import residual_tick
from ternary_tda import TernaryTDAMesh, bitnet_quantize

def process_manifold(close, high, low):
    try:
        import importlib
        engine = importlib.import_module("src.stocksnode.core.financial_tensor").Web3FinancialTensor()
        out = engine.process_manifold(close, high, low)
        out["engine"] = "stocksnode.Web3FinancialTensor"
        return out
    except Exception:
        means = np.mean(close, axis=1, keepdims=True)
        stds = np.maximum(np.std(close, axis=1, keepdims=True), 1e-8)
        z = (close - means) / stds
        rets = np.diff(close, axis=1) / np.maximum(close[:, :-1], 1e-8)
        base = np.std(rets, axis=1)
        recent = np.std(rets[:, -10:], axis=1) if rets.shape[1] > 10 else base
        q = 1.0 - np.exp(-np.abs(z[:, -1]) * (recent / np.maximum(base, 0.01)))
        turtle = (close[:, -1] - low[:, -1]) / (high[:, -1] - low[:, -1] + 1e-8)
        return {"spot": close[:, -1], "z_score": z[:, -1], "q_mark": q,
                "turtle_alignment": turtle, "engine": "local-numpy-twin"}

def log_returns(close):
    return np.log(np.maximum(close[:, 1:], 1e-8) / np.maximum(close[:, :-1], 1e-8))

def outlook(book=None):
    book = book or load_book()
    close, high, low = book["close"], book["high"], book["low"]
    features = process_manifold(close, high, low)
    rets = log_returns(close)
    q, gamma = bitnet_quantize(rets)
    energy = np.array([s == "energy" for s in book["sectors"]])
    macro = np.array([s == "macro" for s in book["sectors"]])
    tick = residual_tick(rets, mesh=TernaryTDAMesh(drift_threshold=0.12), energy=0.90)
    er, mr = rets[energy], rets[macro]
    return {
        "live": bool(book.get("live")), "source": book.get("source"), "engine": features["engine"],
        "tickers": book["tickers"], "sectors": book["sectors"], "bars": int(close.shape[1]),
        "q_mark": {n: float(features["q_mark"][i]) for i, n in enumerate(book["tickers"])},
        "spot": {n: float(features["spot"][i]) for i, n in enumerate(book["tickers"])},
        "gamma": float(gamma), "absmean_sparsity": float((np.asarray(q) == 0).mean()),
        "svd_k": tick.get("k"), "retained_energy": tick.get("retained_energy"),
        "energy_vol": float(np.std(er)), "macro_vol": float(np.std(mr)),
        "shift_ratio": float(np.std(mr) / (np.std(er) + 1e-12)),
        "energy_h0_mean_death": h0_persistence(er)["mean_death"],
        "macro_h0_mean_death": h0_persistence(mr)["mean_death"],
        "sparse": compare_formats(rets), "split": "ticker labels — not column halves",
    }

if __name__ == "__main__":
    import json
    print(json.dumps(outlook(), indent=2, default=str))
