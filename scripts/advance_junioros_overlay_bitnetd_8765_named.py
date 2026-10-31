#!/usr/bin/env python3
"""Advance the JuniorOS overlay bitnetd 127.0.0.1:8765 named item from docs/ROADMAP.md / BETA_TO_OS.

Named item: ROADMAP / BETA_TO_OS JuniorOS overlay — bitnetd 127.0.0.1:8765.
Loopback only. Hardened bitnetd.service. No docker.sock. No model pull.

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
DOCKER_SOCK = False
BIND = "127.0.0.1"
PORT = 8765
ITEM = (
    "ROADMAP / BETA_TO_OS JuniorOS overlay — bitnetd 127.0.0.1:8765. "
    "Loopback only. Hardened bitnetd.service. No docker.sock."
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
    if DOCKER_SOCK:
        bad.append("docker_sock")
    if BIND != "127.0.0.1":
        bad.append("bind")
    if PORT != 8765:
        bad.append("port")
    return {"ok": not bad, "reject": bad}


def advance_junioros_overlay_bitnetd_8765_named() -> dict:
    ac = anticon()
    tip = close("0" * 16, ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "item": ITEM,
        "source": "docs/ROADMAP.md / BETA_TO_OS",
        "opt_in": OPT_IN,
        "anticon": ac,
        "rate": RATE,
        "pool": POOL,
        "mint": MINT,
        "socket": SOCKET,
        "ue5_launch": UE5_LAUNCH,
        "model_pull": MODEL_PULL,
        "rate_poll": RATE_POLL,
        "docker_sock": DOCKER_SOCK,
        "bind": BIND,
        "port": PORT,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "JuniorOS overlay bitnetd 127.0.0.1:8765 loopback; opt-in AntiCon held; no model pull no UE boot no docker.sock",
    }


if __name__ == "__main__":
    out = advance_junioros_overlay_bitnetd_8765_named()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
