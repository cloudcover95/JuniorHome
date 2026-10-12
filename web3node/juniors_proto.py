"""Prototype safeguard. Local chain. Not a layer one."""
from __future__ import annotations

import hashlib
import time

from chain_record import run


def pack(text: str) -> bytes:
    raw = text.encode()[:32]
    xs = [(b - 128) / 128.0 for b in raw]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / gamma)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return bytes(t + 1 for t in out)


def bench(n: int = 90) -> dict:
    t0 = time.perf_counter()
    prev = "0" * 16
    for i in range(n):
        body = pack(f"close-{i}|0.05|1")
        prev = hashlib.sha256(prev.encode() + body).hexdigest()[:16]
    us = (time.perf_counter() - t0) * 1e6
    return {"n": n, "us": round(us, 1), "tip": prev, "bytes": n * 16}


def safeguard() -> dict:
    linked = run([("dev", ["gaia"]), ("home", ["os"])])
    bad_prev = linked.get("tip") == "f" * 16
    measured = bench()
    return {
        "ok": bool(linked.get("ok")) and not bad_prev,
        "tip": linked.get("tip"),
        "closes": linked.get("n"),
        "bench": measured,
        "mint": False,
        "socket": False,
        "peers": 0,
        "network": False,
        "layer_one": False,
        "pq_ready": False,
        "sheet_sha": "009e41a",
        "missing": ["peer gossip", "consensus", "ml-dsa verify", "block production"],
    }
