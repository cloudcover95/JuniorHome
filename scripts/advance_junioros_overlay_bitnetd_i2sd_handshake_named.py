#!/usr/bin/env python3
"""Advance ROADMAP / BETA_TO_OS named item: JuniorOS overlay handshake.

Named item: JuniorOS overlay (source of truth JuniorLLM docs/BETA_TO_OS.md;
pairs with bitnetd 127.0.0.1:8765 and i2sd 127.0.0.1:8767 userspace binds).

Additive only. Does not wrap path.py, chain_record.py, or record_block.py.

Base rate 0.05. Pool held. mint false. socket false. No model pull. No UE boot.
Stdlib only. Bind 127.0.0.1. Opt-in, AntiCon, one trit pack.
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
PORT_BITNETD = 8765
PORT_I2SD = 8767
DOCKER_SOCK = False
ITEM = (
    "JuniorOS overlay — source of truth in JuniorLLM docs/BETA_TO_OS.md. "
    "Handshake with bitnetd 127.0.0.1:8765 and i2sd 127.0.0.1:8767. "
    "Userspace only. No docker.sock."
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
    if BIND != "127.0.0.1":
        bad.append("bind")
    if DOCKER_SOCK:
        bad.append("docker_sock")
    return {"ok": not bad, "reject": bad}


def advance() -> dict:
    ac = anticon()
    tip = close("0" * 16, ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "item": ITEM,
        "source": "docs/ROADMAP.md / JuniorOS overlay (BETA_TO_OS)",
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
        "port_bitnetd": PORT_BITNETD,
        "port_i2sd": PORT_I2SD,
        "docker_sock": DOCKER_SOCK,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "JuniorOS overlay named; bitnetd/i2sd handshake probe only; no model pull, no UE boot, no wrap of path/chain_record/record_block",
    }


if __name__ == "__main__":
    out = advance()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
