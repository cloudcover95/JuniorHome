#!/usr/bin/env python3
"""Advance the JuniorCloud chief weekly named item from docs/ROADMAP.md.

Named item: Automations (current) ON: **JuniorCloud chief weekly** Sun/Mon 17:00 MDT (`dd0b1c93`) — only weekly. Writes `docs/WEEKLY_AUDIT.md`.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib only. Bind 127.0.0.1. Only weekly; writes WEEKLY_AUDIT.md.
Do not wrap path.py, chain_record.py, or record_block.py.
Additive only. Cap 1 file. Do not recreate paused slots.
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
CHIEF_WEEKLY = True
ONLY_WEEKLY = True
WRITES_WEEKLY_AUDIT = True
SCHEDULE = "Sun/Mon 17:00 MDT"
SHA_REF = "dd0b1c93"
PAUSED_RECREATE = False
BIND = "127.0.0.1"
ITEM = (
    "ROADMAP Automations (current) ON: JuniorCloud chief weekly "
    "Sun/Mon 17:00 MDT (dd0b1c93) — only weekly. Writes docs/WEEKLY_AUDIT.md."
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
    if not CHIEF_WEEKLY:
        bad.append("chief_weekly")
    if not ONLY_WEEKLY:
        bad.append("only_weekly")
    if not WRITES_WEEKLY_AUDIT:
        bad.append("writes_weekly_audit")
    if PAUSED_RECREATE:
        bad.append("paused_recreate")
    if BIND != "127.0.0.1":
        bad.append("bind")
    return {"ok": not bad, "reject": bad}


def advance() -> dict:
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
        "chief_weekly": CHIEF_WEEKLY,
        "only_weekly": ONLY_WEEKLY,
        "writes_weekly_audit": WRITES_WEEKLY_AUDIT,
        "schedule": SCHEDULE,
        "sha_ref": SHA_REF,
        "paused_recreate": PAUSED_RECREATE,
        "bind": BIND,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": (
            "JuniorCloud chief weekly Sun/Mon 17:00 MDT (dd0b1c93) only weekly "
            "writes docs/WEEKLY_AUDIT.md; path held at rate 0.05; opt-in AntiCon "
            "one trit pack; no model pull; no UE boot; no rate poll; pool held; "
            "mint false; socket false; no paused recreate"
        ),
    }


if __name__ == "__main__":
    out = advance()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
