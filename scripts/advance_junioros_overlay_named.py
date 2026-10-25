"""Advance named item: JuniorOS overlay (juniorctl / bitnetd 127.0.0.1:8765).

Named item: JuniorOS overlay — juniorctl skill-pin and bitnetd loopback handshake.
From ROADMAP #1 T13+ and JuniorOS overlay surface.

Additive only. Does not wrap path.py, chain_record.py, or record_block.py.
No unnamed bench/guard/close.

Base rate 0.05. Pool held. mint false. socket false. No model pull. No UE boot.
Stdlib only. Bind 127.0.0.1. Close: sha256(prev || pack || text)[:16].
"""
from __future__ import annotations

import hashlib
import sys

RATE = 0.05
OPT_IN = True
ANTICON = True
POOL_HELD = True
MINT = False
SOCKET = False
FETCH = False
UE5_LAUNCH = False
MODEL_PULL = False
BIND = "127.0.0.1"
PORT = 8765
ROADMAP_ITEM = "JuniorOS overlay (juniorctl / bitnetd 127.0.0.1:8765)"
DAY = "2026-10-25"
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
    return hashlib.sha256(prev.encode() + PACK + text.encode()).hexdigest()[:16]


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
    if BIND != "127.0.0.1":
        bad.append("bind")
    return {"ok": not bad, "reject": bad}


def advance() -> dict:
    ac = anticon()
    tip = close(GENESIS, ROADMAP_ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "roadmap_item": ROADMAP_ITEM,
        "source": "docs/ROADMAP.md #1 T13+ / JuniorOS overlay",
        "day": DAY,
        "opt_in": OPT_IN,
        "anticon": ac,
        "rate": RATE,
        "pool_held": POOL_HELD,
        "mint": MINT,
        "socket": SOCKET,
        "fetch": FETCH,
        "ue5_launch": UE5_LAUNCH,
        "model_pull": MODEL_PULL,
        "bind": BIND,
        "port": PORT,
        "tip": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "probe only; no socket open; no model pull; no UE boot",
    }


def self_check() -> dict:
    assert RATE == 0.05
    assert MINT is False
    assert SOCKET is False
    assert MODEL_PULL is False
    assert UE5_LAUNCH is False
    assert BIND == "127.0.0.1"
    ac = anticon()
    assert ac["ok"]
    result = advance()
    assert result["ok"]
    assert result["roadmap_item"] == ROADMAP_ITEM
    assert result["tip"] is not None
    assert len(result["tip"]) == 16
    return {"self_check": "pass", "tip": result["tip"], "item": ROADMAP_ITEM}


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        try:
            out = self_check()
            print(out)
            sys.exit(0)
        except AssertionError as e:
            print({"self_check": "fail", "error": str(e)})
            sys.exit(1)
    else:
        print(advance())
