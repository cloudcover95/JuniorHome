"""FieldCore + Flagstaff envelope on the residual tick.

JuniorBitNetFieldCore (JuniorLLM default port) scores crowd/field trits.
Flagstaff is a public municipal node — not a private-land pin.
stocksnode Web3FinancialTensor builds the manifold when importable.
Residual tick / SVD retain stay the geometry path.
JuniorMemSys gets the JSONL row. AGI_SDK ModelRouter is named, not vendored.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from svd_residual_tick import residual_tick
from ternary_tda import TernaryTDAMesh

FLAGSTAFF = {
    "id": "flagstaff",
    "name": "Flagstaff FieldCore",
    "region": "Colorado Plateau / northern Arizona",
    "tenure": "municipal_public",
    "lat": 35.1983,
    "lon": -111.6513,
    "precision": "city_centroid",
}

FIELDCORE_LABELS = (
    "beta_trust",
    "access_caution",
    "tenure_usfs",
    "tenure_private",
    "nav_priority",
)

JUNIORLLM_FIELD_PORT = "JuniorBitNetFieldCore"
AGI_ROUTE_HW = "apple_silicon"


def fieldcore_scores(seed: int = 13) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    raw = rng.random(len(FIELDCORE_LABELS))
    raw = raw / raw.sum()
    return {name: float(v) for name, v in zip(FIELDCORE_LABELS, raw)}


def pick_juniorllm_port(task: str) -> str:
    t = (task or "").lower()
    if any(k in t for k in ("cad", "dxf", "omega", "draft")):
        return "JuniorBitNetDraft"
    if "astra" in t or "reason" in t:
        return "JuniorAstraReason"
    if "safety" in t or "fable" in t:
        return "JuniorFable"
    return JUNIORLLM_FIELD_PORT


def stocksnode_manifold(n: int = 16, t: int = 32, seed: int = 5) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.normal(size=(n, t)), axis=1) + 100.0
    high = close + rng.random((n, t))
    low = close - rng.random((n, t))
    try:
        import importlib

        mod = importlib.import_module("src.stocksnode.core.financial_tensor")
        engine = mod.Web3FinancialTensor()
        out = engine.process_manifold(close, high, low)
        out["close"] = close
        return out
    except Exception:
        means = np.mean(close, axis=1, keepdims=True)
        stds = np.maximum(np.std(close, axis=1, keepdims=True), 1e-8)
        z = (close - means) / stds
        rets = np.diff(close, axis=1) / np.maximum(close[:, :-1], 1e-8)
        base = np.std(rets, axis=1)
        recent = np.std(rets[:, -10:], axis=1)
        q = 1.0 - np.exp(-np.abs(z[:, -1]) * (recent / np.maximum(base, 0.01)))
        turtle = (close[:, -1] - low[:, -1]) / (high[:, -1] - low[:, -1] + 1e-8)
        return {
            "spot": close[:, -1],
            "z_score": z[:, -1],
            "q_mark": q,
            "turtle_alignment": turtle,
            "close": close,
        }


def manifold_matrix(features: dict[str, Any]) -> np.ndarray:
    if "close" in features:
        return np.asarray(features["close"], dtype=np.float64)
    cols = [
        np.asarray(features["spot"], dtype=np.float64),
        np.asarray(features["z_score"], dtype=np.float64),
        np.asarray(features["q_mark"], dtype=np.float64),
        np.asarray(features["turtle_alignment"], dtype=np.float64),
    ]
    return np.stack(cols, axis=1)


def memsys_append(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def run_flagstaff(task: str = "field flagstaff", seed: int = 13) -> dict[str, Any]:
    port = pick_juniorllm_port(task)
    labels = fieldcore_scores(seed)
    features = stocksnode_manifold(seed=seed)
    matrix = manifold_matrix(features)
    mesh = TernaryTDAMesh(drift_threshold=0.12)
    tick = residual_tick(matrix, mesh=mesh, energy=0.95)
    row = {
        "t": int(time.time() * 1000),
        "node": FLAGSTAFF,
        "fieldcore": labels,
        "juniorllm_port": port,
        "agi_sdk": {"router": "ModelRouter", "hardware": AGI_ROUTE_HW, "precision": "ternary"},
        "memsys": "JuniorMemSys-Suite jsonl palace row",
        "stocksnode": {
            "q_mark_mean": float(np.mean(features["q_mark"])),
            "spot_mean": float(np.mean(features["spot"])),
        },
        "tick": {
            "k": tick.get("k"),
            "retained_energy": tick.get("retained_energy"),
            "drift": tick.get("drift"),
            "qmark_collapse": tick.get("qmark_collapse"),
            "cpu_intent": tick.get("cpu_intent"),
        },
    }
    out = Path(__file__).resolve().parent / "fieldcore_flagstaff.jsonl"
    memsys_append(row, out)
    row["memsys_path"] = str(out)
    return row


if __name__ == "__main__":
    result = run_flagstaff()
    print("[port]", result["juniorllm_port"])
    print("[node]", result["node"]["id"], result["node"]["tenure"])
    print("[fieldcore]", {k: round(v, 3) for k, v in result["fieldcore"].items()})
    print("[stocksnode]", result["stocksnode"])
    print("[tick]", result["tick"])
    print("[agi]", result["agi_sdk"])
    print("[memsys]", result["memsys_path"])
