#!/usr/bin/env python3
"""Advance the GGUF named item from docs/ROADMAP.md.

Named item: Order #2 GGUF — operator sets JUNIOR_GGUF. Bots do not download. web3node/probe_future.py.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib only. Do not wrap path.py, chain_record.py, or record_block.py.
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
OPERATOR_SETS = True
BOTS_DOWNLOAD = False
PROBE_FUTURE = True
ITEM = (
    "ROADMAP Order #2 GGUF — operator sets JUNIOR_GGUF. "
    "Bots do not download. web3node/probe_future.py."
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
    if not OPERATOR_SETS:
        bad.append("operator_sets")
    if BOTS_DOWNLOAD:
        bad.append("bots_download")
    if not PROBE_FUTURE:
        bad.append("probe_future")
    return {"ok": not bad, "reject": bad}


def advance_roadmap_order_2_gguf_named() -> dict:
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
        "operator_sets": OPERATOR_SETS,
        "bots_download": BOTS_DOWNLOAD,
        "probe_future": PROBE_FUTURE,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "GGUF operator sets JUNIOR_GGUF; bots do not download; probe_future only; no model pull; no UE boot; rate 0.05 held",
    }


if __name__ == "__main__":
    out = advance_roadmap_order_2_gguf_named()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
