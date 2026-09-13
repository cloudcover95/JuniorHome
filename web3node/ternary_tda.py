"""Ternary sidecar next to svd_metal.py.

SVD / RSVD stay the geometry path (mx.linalg.svd on Metal/CPU).
This module AbsMean-quantizes a state to W_q in {-1, 0, 1},
measures identity drift on the discrete mesh, and can emit
FrameForge CPU intent logits. It does not scale knockback.

Gemini V305 claimed SVD was deprecated and flushed Parquet to /vault.
Neither happens here. Optional JSONL is local. giotto-tda stays in tda_mesh.py.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

try:
    import mlx.core as mx  # type: ignore

    BACKEND = "mlx"
except ImportError:  # loopback / CI without Metal
    import numpy as mx  # type: ignore

    BACKEND = "numpy"

ROSTER = ("Vesper", "Quill", "Relay", "Forge")
EPS = 1e-7


def _item(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _norm(matrix: Any) -> Any:
    if BACKEND == "mlx":
        return mx.linalg.norm(matrix)
    return mx.linalg.norm(matrix)


def bitnet_quantize(weight: Any) -> tuple[Any, float]:
    """AbsMean ternary: W_q = clip(round(W / gamma), -1, 1), gamma = mean(|W|)."""
    gamma = mx.mean(mx.abs(weight))
    gamma_f = _item(gamma)
    scaled = weight / (gamma + EPS)
    quantized = mx.clip(mx.round(scaled), -1.0, 1.0)
    return quantized, gamma_f


def identity_drift(current: Any, previous: Any | None) -> tuple[float, Any]:
    """1 - <Y_t, Y_{t-1}> / (||Y_t|| ||Y_{t-1}||) on ternary states."""
    if previous is None:
        return 0.0, current
    num = mx.sum(current * previous)
    denom = _norm(current) * _norm(previous)
    drift = 1.0 - (num / (denom + EPS))
    return _item(drift), current


def cpu_intent_logits(quantized: Any, drift: float) -> dict[str, float]:
    """Four roster logits from ternary mass + drift. Not launch scale."""
    flat = quantized.reshape(-1)
    if BACKEND == "mlx":
        neg = _item(mx.mean(flat < 0))
        zero = _item(mx.mean(flat == 0))
        pos = _item(mx.mean(flat > 0))
    else:
        neg = float((flat < 0).mean())
        zero = float((flat == 0).mean())
        pos = float((flat > 0).mean())
    raw = {
        "Vesper": pos + 0.15 * drift,
        "Quill": zero + 0.05 * drift,
        "Relay": 0.5 * (pos + neg),
        "Forge": neg + 0.20 * drift,
    }
    peak = max(raw.values())
    exp = {name: math.exp(score - peak) for name, score in raw.items()}
    z = sum(exp.values()) or 1.0
    return {name: exp[name] / z for name in ROSTER}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


class TernaryTDAMesh:
    """Stateful tick. Sidecar — does not replace compute_hardware_svd."""

    def __init__(
        self,
        drift_threshold: float = 0.15,
        telemetry_path: str | Path | None = None,
    ) -> None:
        self.drift_threshold = drift_threshold
        self.previous = None
        self.telemetry_path = (
            Path(telemetry_path)
            if telemetry_path
            else Path(__file__).resolve().parent / "telemetry.jsonl"
        )

    def tick(self, raw: Any) -> dict[str, Any]:
        quantized, gamma = bitnet_quantize(raw)
        drift, self.previous = identity_drift(quantized, self.previous)
        collapse = drift > self.drift_threshold
        logits = cpu_intent_logits(quantized, drift)
        row = {
            "t": int(time.time() * 1000),
            "backend": BACKEND,
            "gamma": gamma,
            "drift": drift,
            "qmark_collapse": collapse,
            "cpu_intent": logits,
            "shape": list(getattr(raw, "shape", [])),
        }
        append_jsonl(self.telemetry_path, row)
        return row


def _demo_matrix(dim: int = 32):
    if BACKEND == "mlx":
        return mx.random.normal((dim, dim))
    rng = mx.random.default_rng(42)
    return rng.normal(size=(dim, dim)).astype("float32")


if __name__ == "__main__":
    print(f"[ternary_tda] backend={BACKEND}")
    core = TernaryTDAMesh(drift_threshold=0.10)
    first = core.tick(_demo_matrix())
    print("[tick1]", {k: first[k] for k in ("drift", "gamma", "qmark_collapse")})
    second_raw = _demo_matrix()
    if BACKEND == "mlx":
        second_raw = second_raw + mx.random.normal(second_raw.shape) * 0.4
    else:
        second_raw = second_raw + mx.random.default_rng(7).normal(size=second_raw.shape) * 0.4
    second = core.tick(second_raw)
    print("[tick2]", {k: second[k] for k in ("drift", "gamma", "qmark_collapse")})
    print("[cpu_intent]", second["cpu_intent"])
    print("[telemetry]", core.telemetry_path)
