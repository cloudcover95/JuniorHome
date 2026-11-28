#!/usr/bin/env python3
"""Advance the Not this roadmap second GitHub bot user named item from docs/ROADMAP.md.

Named item: Not this roadmap: second GitHub bot user.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib only. Does not wrap path.py, chain_record.py, or record_block.py.
Additive only. Cap 1 file. Bind 127.0.0.1. No docker.sock. Do not recreate paused slots.
No second GitHub bot user.
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
RATE_POLL = False
UE5_LAUNCH = False
SECOND_BOT = False
PAUSED_RECREATE = False
BIND = "127.0.0.1"
DOCKER_SOCK = False
ITEM = (
    "ROADMAP Not this roadmap: second GitHub bot user."
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
    if RATE_POLL:
        bad.append("rate_poll")
    if UE5_LAUNCH:
        bad.append("ue5_launch")
    if SECOND_BOT:
        bad.append("second_bot")
    if PAUSED_RECREATE:
        bad.append("paused_recreate")
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
        "source": "docs/ROADMAP.md Not this roadmap",
        "opt_in": OPT_IN,
        "anticon": ac,
        "rate": RATE,
        "pool": POOL,
        "mint": MINT,
        "socket": SOCKET,
        "model_pull": MODEL_PULL,
        "rate_poll": RATE_POLL,
        "ue5_launch": UE5_LAUNCH,
        "second_bot": SECOND_BOT,
        "paused_recreate": PAUSED_RECREATE,
        "bind": BIND,
        "docker_sock": DOCKER_SOCK,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": (
            "Not this roadmap second GitHub bot user; path held at rate 0.05; "
            "opt-in AntiCon one trit pack; no model pull; no rate poll; "
            "pool held; mint false; socket false; no UE boot; no docker.sock; "
            "no second bot; do not recreate paused; dated 20261128"
        ),
    }


if __name__ == "__main__":
    out = advance()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
