#!/usr/bin/env python3
"""Advance the JuniorOS overlay named item.

Named item: JuniorOS overlay — local-first edge home, 45 W envelope, loopback only.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No UE boot.
Stdlib only. Does not wrap path.py, chain_record.py, or record_block.py.
Additive only. Cap 1 file. Default FrameForge2D.
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
MODEL_PULL = False
UE5_BOOT = False
OVERLAY = True
LOOPBACK = "127.0.0.1"
ITEM = (
    "JuniorOS overlay — local-first edge home on 45 W envelope. Loopback only. Default FrameForge2D. No model pull. No UE boot."
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
    if MODEL_PULL:
        bad.append("model_pull")
    if UE5_BOOT:
        bad.append("ue5_boot")
    if not OVERLAY:
        bad.append("overlay")
    if LOOPBACK != "127.0.0.1":
        bad.append("loopback")
    return {"ok": not bad, "reject": bad}


def advance_junioros_overlay_named() -> dict:
    ac = anticon()
    tip = close("0" * 16, ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "item": ITEM,
        "source": "JuniorOS overlay",
        "opt_in": OPT_IN,
        "anticon": ac,
        "rate": RATE,
        "pool": POOL,
        "mint": MINT,
        "socket": SOCKET,
        "model_pull": MODEL_PULL,
        "ue5_boot": UE5_BOOT,
        "overlay": OVERLAY,
        "loopback": LOOPBACK,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": (
            "JuniorOS overlay held: local-first 45 W; loopback 127.0.0.1; "
            "path held at rate 0.05; opt-in AntiCon one trit pack; "
            "pool held; mint false; socket false; no model pull; no UE boot"
        ),
    }


if __name__ == "__main__":
    out = advance_junioros_overlay_named()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
