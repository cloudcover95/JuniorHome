"""ROADMAP #2 GGUF — operator sets JUNIOR_GGUF. Bots do not download.

Advances JuniorHome docs/ROADMAP.md item 2. Opt-in AntiCon one trit pack,
base rate 0.05, pool held, mint false, socket false. No poll sets the rate.
No model pull. No download. No UE boot. Stdlib only. Close on held tip.
"""
from __future__ import annotations

import hashlib
import os

RATE = 0.05
OPT_IN = True
ANTICON = True
POOL = "held"
MINT = False
SOCKET = False
DOWNLOAD = False
MODEL_PULL = False
UE5 = False
BIND = "127.0.0.1"
ITEM = "ROADMAP #2 GGUF — operator sets JUNIOR_GGUF. Bots do not download."
ENV_KEY = "JUNIOR_GGUF"
TRIT_PACK = (1, 0, -1, 1, 0, -1)

_ENC = {0: 0, 1: 1, -1: 2}


def pack(trits) -> bytes:
    acc = bits = 0
    out = bytearray()
    for t in trits:
        if t not in _ENC:
            raise ValueError("trit must be -1, 0 or 1")
        acc = (acc << 2) | _ENC[t]
        bits += 2
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


PACK = pack(TRIT_PACK)


def close(prev: str, text: str) -> str:
    body = f"{prev}|{text}|{RATE}".encode() + PACK
    return hashlib.sha256(body).hexdigest()[:16]


def anticon(env: dict) -> dict:
    bad = []
    if not OPT_IN:
        bad.append("opt_in")
    if env.get("download") or DOWNLOAD:
        bad.append("download")
    if env.get("model_pull") or MODEL_PULL:
        bad.append("model_pull")
    if env.get("socket") or SOCKET:
        bad.append("socket")
    if env.get("mint") or MINT:
        bad.append("mint")
    if env.get("ue5") or UE5:
        bad.append("ue5")
    if env.get("rate") and env["rate"] != RATE:
        bad.append("rate")
    if env.get("bind") and env["bind"] != BIND:
        bad.append("bind")
    return {"ok": not bad, "reject": bad}


def receipt(prev: str = "0" * 16, env: dict | None = None) -> dict:
    env = env or {}
    ac = anticon(env)
    raw = (os.environ.get(ENV_KEY) or "").strip()
    set_by_op = bool(raw)
    tip = close(prev, ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "item": ITEM,
        "opt_in": OPT_IN,
        "anticon": ac,
        "rate": RATE,
        "pool": POOL,
        "mint": MINT,
        "socket": SOCKET,
        "download": DOWNLOAD,
        "model_pull": MODEL_PULL,
        "ue5": UE5,
        "bind": BIND,
        "env_set": set_by_op,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
    }


def selftest() -> int:
    row = receipt()
    assert row["ok"] and row["rate"] == 0.05
    assert row["pool"] == "held" and row["mint"] is False and row["socket"] is False
    assert row["download"] is False and row["model_pull"] is False
    assert "ROADMAP #2 GGUF" in row["item"]
    assert "do not download" in row["item"]
    bad = receipt(env={"download": True})
    assert not bad["ok"] and "download" in bad["anticon"]["reject"]
    rate_bad = receipt(env={"rate": 0.1})
    assert not rate_bad["ok"]
    return 0


if __name__ == "__main__":
    import json
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print(json.dumps(receipt(), indent=2))
