"""Wallet class catalog. Trit pack of the name. No seed, no send."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "wallet_class.jsonl"

# injected: user approves in their own app. custodial: no local signer.
CLASSES = {
    "metamask": "injected",
    "phantom": "injected",
    "mew": "injected",
    "rabby": "injected",
    "uphold": "custodial",
    "cashapp": "custodial",
    "xmoney": "custodial",
}


def pack(name: str) -> dict:
    kind = CLASSES.get(name, "unknown")
    xs = [((ord(c) % 5) - 2) / 2.0 for c in name[:32]] or [0.0]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    trits = []
    for x in xs:
        q = round(x / gamma)
        trits.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return {
        "name": name,
        "class": kind,
        "trits": trits,
        "sha3": hashlib.sha3_256(name.encode()).hexdigest()[:16],
        "seed": False,
        "send": False,
        "address": False,
        "rpc": False,
    }


def bench(n: int = 2000) -> dict:
    t0 = time.perf_counter()
    for _ in range(n):
        for name in CLASSES:
            pack(name)
    ms = (time.perf_counter() - t0) / (n * len(CLASSES)) * 1000.0
    rows = [pack(name) for name in CLASSES]
    body = {"ms_per_note": round(ms, 4), "n": len(rows), "rows": rows, "model_pull": False}
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ms": body["ms_per_note"], "n": body["n"]}) + "\n")
    return body
