"""One Home tick: Winsor → trit. Stdlib. Bind never 0.0.0.0."""
from __future__ import annotations

import json
import time
from math import floor
from pathlib import Path


def pack(weights: list[float], percentile: float = 95.0) -> dict:
    abs_w = [abs(float(w)) for w in weights]
    if not abs_w:
        return {"ok": False}
    s = sorted(abs_w)
    tau = s[min(len(s) - 1, max(0, floor((percentile / 100.0) * (len(s) - 1))))]
    winsor = [min(a, tau) for a in abs_w]
    gamma = sum(winsor) / len(winsor)
    g = gamma + 1e-7
    trit = [max(-1, min(1, int(round(float(w) / g)))) for w in weights]
    return {"ok": True, "n": len(trit), "tau": tau, "gamma": gamma,
            "sparsity": sum(1 for t in trit if t == 0) / len(trit), "trit": trit}


def tick(note: str, vault: Path) -> dict:
    xs = [float(ord(c) % 97) for c in (note or "x")[:32]]
    t0 = time.perf_counter()
    q = pack(xs)
    dt_us = (time.perf_counter() - t0) * 1e6
    row = {"note": note[:80], "us": round(dt_us, 3), "gamma": q.get("gamma"),
           "sparsity": q.get("sparsity"), "bind": "127.0.0.1"}
    vault.mkdir(parents=True, exist_ok=True)
    log = vault / "trit_tick.jsonl"
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    row["ok"] = bool(q.get("ok"))
    return row


if __name__ == "__main__":
    print(json.dumps(tick("buy oats and fix the porch light", Path("/tmp/juniorhome_vault")), indent=2))
