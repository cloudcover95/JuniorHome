#!/usr/bin/env python3
"""Advance the named item UE5 launched from docs/ROADMAP.md.

Named item: UE5 launched — JUNIOR_UE5=1 and watts>=80. Default FrameForge2D. os_route.launch false until then.
(from docs/ROADMAP.md Order item 5).

Additive only. Does not wrap path.py, chain_record.py, or record_block.py.
Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
Do not boot UE. Bind 127.0.0.1. Stdlib only.
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
BIND = "127.0.0.1"
ITEM = (
    "ROADMAP Order (do not skip) #5 UE5 launched — JUNIOR_UE5=1 and watts>=80. "
    "Default FrameForge2D. os_route.launch false until then."
)
GENESIS = "0" * 16
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
    if BIND != "127.0.0.1":
        bad.append("bind")
    return {"ok": not bad, "reject": bad}


def advance() -> dict:
    ac = anticon()
    tip = close(GENESIS, ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "item": ITEM,
        "source": "docs/ROADMAP.md Order #5 UE5 launched",
        "opt_in": OPT_IN,
        "anticon": ac,
        "rate": RATE,
        "pool": POOL,
        "mint": MINT,
        "socket": SOCKET,
        "ue5_launch": UE5_LAUNCH,
        "model_pull": MODEL_PULL,
        "rate_poll": RATE_POLL,
        "bind": BIND,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "UE5 launched named confirm; JUNIOR_UE5=1 and watts>=80; default FrameForge2D; os_route.launch false; do not boot UE; no wrap of path/chain_record/record_block",
    }


def self_check() -> dict:
    assert RATE == 0.05
    assert OPT_IN and ANTICON
    assert POOL == "held"
    assert not (MINT or SOCKET or UE5_LAUNCH or MODEL_PULL or RATE_POLL)
    assert BIND == "127.0.0.1"
    assert "UE5 launched" in ITEM
    assert "JUNIOR_UE5=1" in ITEM
    assert "FrameForge2D" in ITEM
    out = advance()
    assert out["ok"] and out["advanced"] and out["hash"] and len(out["hash"]) == 16
    assert close(GENESIS, ITEM) == out["hash"]
    return out


if __name__ == "__main__":
    try:
        result = self_check()
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except AssertionError as e:
        print(json.dumps({"ok": False, "error": repr(e)}))
        sys.exit(1)
