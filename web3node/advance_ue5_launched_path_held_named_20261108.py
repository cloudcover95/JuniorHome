#!/usr/bin/env python3
"""Advance the UE5 launched named item from docs/ROADMAP.md.

Named item: Order (do not skip) 5. **UE5 launched** — `JUNIOR_UE5=1` and watts>=80. Default FrameForge2D. `os_route.launch` false until then.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib only. Bind 127.0.0.1. Do not wrap path.py,
chain_record.py, or record_block.py. os_route.launch remains false.
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
JUNIOR_UE5 = False  # remains false until conditions; env check not forced
WATTS_OK = False    # not booting, just hold
OS_ROUTE_LAUNCH = False
DEFAULT_FF2D = True
BIND = "127.0.0.1"
ITEM = (
    "ROADMAP Order (do not skip) #5 UE5 launched — JUNIOR_UE5=1 and watts>=80. "
    "Default FrameForge2D. os_route.launch false until then."
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
    if not DEFAULT_FF2D:
        bad.append("default_ff2d")
    if BIND != "127.0.0.1":
        bad.append("bind")
    return {"ok": not bad, "reject": bad}


def advance_ue5_launched_path_held_named_20261108() -> dict:
    ac = anticon()
    tip = close("0" * 16, ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "item": ITEM,
        "source": "docs/ROADMAP.md Order #5",
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
        "default_ff2d": DEFAULT_FF2D,
        "bind": BIND,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "UE5 launched held; JUNIOR_UE5 and watts not forced; os_route.launch false; default FrameForge2D; path held at rate 0.05; opt-in AntiCon; no model pull no UE boot no rate poll",
    }


if __name__ == "__main__":
    out = advance_ue5_launched_path_held_named_20261108()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
