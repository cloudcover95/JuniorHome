"""Advance the T13+ named item from docs/ROADMAP.md.

Named item: ROADMAP Order #1 T13+ — juniorctl skill-pin height. Overnight owns this.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib only. Bind 127.0.0.1. Do not wrap path.py, chain_record.py,
or record_block.py. Probe only; no juniorctl boot.
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
OVERNIGHT_OWNS = True
SKILL_PIN = True
ITEM = (
    "ROADMAP Order #1 T13+ — juniorctl skill-pin height. Overnight owns this."
)
SOURCE = "docs/ROADMAP.md"
TRIT_PACK = (1, 0, -1, 1, 0, -1)
GENESIS = "0" * 16

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
    if not OVERNIGHT_OWNS:
        bad.append("overnight_owns")
    if not SKILL_PIN:
        bad.append("skill_pin")
    return {"ok": not bad, "reject": bad}


def advance_t13_skill_pin() -> dict:
    ac = anticon()
    tip = close(GENESIS, ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "item": ITEM,
        "source": SOURCE,
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
        "overnight_owns": OVERNIGHT_OWNS,
        "skill_pin": SKILL_PIN,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "T13+ juniorctl skill-pin height; overnight owns; probe only; no model pull; no UE boot",
    }


if __name__ == "__main__":
    out = advance_t13_skill_pin()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
