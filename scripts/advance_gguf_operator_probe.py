#!/usr/bin/env python3
"""Advance the GGUF named item from docs/ROADMAP.md.

Named item: GGUF — operator sets JUNIOR_GGUF. Bots do not download. web3node/probe_future.py.
From ROADMAP #2. Order do not skip.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib only. Operator sets JUNIOR_GGUF. Bots do not download.
Probe only. No wrap of path/chain_record/record_block.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

RATE = 0.05
OPT_IN = True
ANTICON = True
POOL = "held"
MINT = False
SOCKET = False
UE5_LAUNCH = False
MODEL_PULL = False
RATE_POLL = False
BOTS_DOWNLOAD = False
ITEM = (
    "ROADMAP #2 GGUF — operator sets JUNIOR_GGUF. Bots do not download. "
    "web3node/probe_future.py."
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
    if BOTS_DOWNLOAD:
        bad.append("bots_download")
    return {"ok": not bad, "reject": bad}


def probe_gguf() -> dict:
    gguf = os.environ.get("JUNIOR_GGUF", "")
    exists = bool(gguf) and Path(gguf).is_file()
    return {
        "gguf_env": bool(gguf),
        "gguf_exists": exists,
        "operator_sets": True,
        "bots_do_not_download": True,
        "probe_only": True,
    }


def advance_gguf_operator_probe() -> dict:
    ac = anticon()
    p = probe_gguf()
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
        "bots_download": BOTS_DOWNLOAD,
        "probe": p,
        "trit_pack": list(TRIT_PACK),
        "pack": PACK.hex(),
        "tip": tip,
        "advanced": ac["ok"],
    }


if __name__ == "__main__":
    out = advance_gguf_operator_probe()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
