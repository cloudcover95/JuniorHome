"""BitNet tritquant. stdlib. Does not replace rails/linux/absmean.c."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "tritquant.jsonl"


def winsor(xs: list[float], p: float = 0.95) -> list[float]:
    if not xs:
        return []
    s = sorted(abs(x) for x in xs)
    cap = s[min(len(s) - 1, int(p * (len(s) - 1)))]
    return [max(-cap, min(cap, x)) for x in xs]


def absmean(xs: list[float]) -> tuple[list[int], float]:
    if not xs:
        return [], 1.0
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / gamma)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return out, gamma


def absmax_act(xs: list[float]) -> tuple[list[int], float]:
    peak = max((abs(x) for x in xs), default=1.0) or 1.0
    out = []
    for x in xs:
        q = round(x * 127.0 / peak)
        out.append(127 if q > 127 else (-127 if q < -127 else int(q)))
    return out, peak


def bitdot(x: list[float], w: list[float]) -> dict:
    wq, gamma = absmean(w)
    xq, _ = absmax_act(x)
    n = min(len(xq), len(wq))
    acc = sum(xq[i] * wq[i] for i in range(n))
    return {"y": acc * (gamma / 127.0), "gamma": gamma, "wq": wq, "n": n, "zeros": wq.count(0)}


def pack2(trits: list[int]) -> bytes:
    acc = 0
    n = 0
    out = bytearray()
    for t in trits:
        c = 0 if t == 0 else (1 if t == 1 else 2)
        acc = (acc << 2) | c
        n += 2
        if n == 8:
            out.append(acc)
            acc = 0
            n = 0
    if n:
        out.append(acc << (8 - n))
    return bytes(out)


def pack5(trits: list[int]) -> bytes:
    acc = 0
    n = 0
    out = bytearray()
    for t in trits:
        c = t + 1
        acc = acc * 3 + c
        n += 1
        if n == 5:
            out.append(acc)
            acc = 0
            n = 0
    if n:
        out.append(acc)
    return bytes(out)


def unpack5(buf: bytes, n: int) -> list[int]:
    out: list[int] = []
    for b in buf:
        digits = []
        v = b
        for _ in range(5):
            digits.append(v % 3)
            v //= 3
        for d in reversed(digits):
            if len(out) < n:
                out.append(d - 1)
    return out[:n]


def ticket(x: list[float], w: list[float]) -> dict:
    t0 = time.perf_counter()
    row = bitdot(winsor(x), winsor(w))
    raw = bytes(t + 1 for t in row["wq"])
    body = {
        "protocol": "goldend-osai-omega/1",
        "y": row["y"],
        "gamma": row["gamma"],
        "n": row["n"],
        "zeros": row["zeros"],
        "pack2": pack2(row["wq"]).hex(),
        "pack5": pack5(row["wq"]).hex(),
        "sha3": hashlib.sha3_256(raw).hexdigest()[:16],
        "ms": round((time.perf_counter() - t0) * 1000.0, 4),
        "model_pull": False,
        "mlx": False,
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"sha3": body["sha3"], "ms": body["ms"], "n": body["n"]}) + "\n")
    return body
