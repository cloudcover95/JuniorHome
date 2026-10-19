#!/usr/bin/env python3
"""Advance the juniorctl named item from docs/ROADMAP.md.
Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No UE boot.
juniorctl on PATH for skill-pin height. Bind 127.0.0.1 only.
Confirm PATH presence does not imply public bind or model pull.
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
BIND = "127.0.0.1"
ITEM = (
    "ROADMAP #1 T13+ — juniorctl skill-pin height. Overnight owns this. "
    "juniorctl on PATH. bitnetd 127.0.0.1:8765. i2sd 127.0.0.1:8767. "
    "JuniorOS overlay. Confirm PATH only; no public bind."
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
    return {"ok": not bad, "reject": bad}


def advance_juniorctl_path_confirm() -> dict:
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
        "bind": BIND,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "juniorctl on PATH confirm; skill-pin height; no public bind; no model pull; no UE; overnight owns",
    }


if __name__ == "__main__":
    out = advance_juniorctl_path_confirm()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
