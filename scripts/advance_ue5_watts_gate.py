"""Advance the UE5 watts gate named item from docs/ROADMAP.md.

Named item: ROADMAP #5 UE5 launched — JUNIOR_UE5=1 and watts>=80.
Default FrameForge2D. os_route.launch false until then.
No UE boot on the 45 W node.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Bind 127.0.0.1. Do not wrap path.py, chain_record.py, or record_block.py.
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
DEFAULT_2D = True
JUNIOR_UE5 = False  # remains false until operator sets and watts>=80
WATTS_OK = False    # 45 W node; gate not open
BIND = "127.0.0.1"
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
    if not DEFAULT_2D:
        bad.append("default_2d")
    if JUNIOR_UE5:
        bad.append("junior_ue5")
    if WATTS_OK:
        bad.append("watts_ok")
    if BIND != "127.0.0.1":
        bad.append("bind")
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
        "default_2d": DEFAULT_2D,
        "junior_ue5": JUNIOR_UE5,
        "watts_ok": WATTS_OK,
        "bind": BIND,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "ROADMAP #5 watts gate held; JUNIOR_UE5 remains false; os_route.launch false; no UE boot; no model pull",
    }


if __name__ == "__main__":
    out = advance_ue5_watts_gate()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
