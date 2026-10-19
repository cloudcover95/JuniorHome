"""Advance the UE5 launched named item from docs/ROADMAP.md.

Named item: ROADMAP #5 UE5 launched — JUNIOR_UE5=1 and watts>=80.
Default FrameForge2D. os_route.launch false until then.
No UE boot. Probe only. Stdlib only.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
Bind 127.0.0.1. bitnetd 127.0.0.1:8765, i2sd 127.0.0.1:8767 named.
"""
from __future__ import annotations

import hashlib
import json
import sys

RATE = 0.05
OPT_IN = True
ANTICON = True
POOL = "held"
MINT = False
SOCKET = False
UE5_LAUNCH = False
MODEL_PULL = False
RATE_POLL = False
OS_ROUTE_LAUNCH = False
BIND = "127.0.0.1"
BITNETD_PORT = 8765
I2SD_PORT = 8767
ITEM = (
    "ROADMAP #5 UE5 launched — JUNIOR_UE5=1 and watts>=80. "
    "Default FrameForge2D. os_route.launch false until then. "
    "No UE boot."
)
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


def anticon() -> dict:
    bad = []
    if not OPT_IN:
        bad.append("opt_in")
    if MINT:
        bad.append("mint")
    if SOCKET:
        bad.append("socket")
    if UE5_LAUNCH:
        bad.append("ue5_launch")
    if MODEL_PULL:
        bad.append("model_pull")
    if RATE_POLL:
        bad.append("rate_poll")
    if OS_ROUTE_LAUNCH:
        bad.append("os_route_launch")
    return {"ok": not bad, "reject": bad}


def advance_ue5_watts_gate() -> dict:
    ac = anticon()
    tip = close("0" * 16, ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "item": ITEM,
        "source": "docs/ROADMAP.md",
        "opt_in": OPT_IN,
        "anticon": ac,
        "rate": RATE,
        "pool": POOL,
        "mint": MINT,
        "socket": SOCKET,
        "ue5_launch": UE5_LAUNCH,
        "model_pull": MODEL_PULL,
        "rate_poll": RATE_POLL,
        "os_route_launch": OS_ROUTE_LAUNCH,
        "bind": BIND,
        "bitnetd": f"{BIND}:{BITNETD_PORT}",
        "i2sd": f"{BIND}:{I2SD_PORT}",
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "ROADMAP #5 UE5 watts>=80 gate; default FrameForge2D; os_route.launch false; no boot",
    }


if __name__ == "__main__":
    out = advance_ue5_watts_gate()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
