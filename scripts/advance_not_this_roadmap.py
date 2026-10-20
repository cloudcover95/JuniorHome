"""Advance the Not this roadmap named item from docs/ROADMAP.md.

Named item: Not this roadmap — 0.0.0.0 SaaS, 4B pulls, MP/KAYA, second GitHub bot user.
Boundary confirmation. No SaaS bind. No large model pull. No second bot user.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib only. Bind 127.0.0.1. Do not wrap path/chain_record/record_block.
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
SAAS_BIND = False
FOUR_B_PULL = False
SECOND_BOT_USER = False
BIND = "127.0.0.1"
ITEM = (
    "ROADMAP Not this roadmap: 0.0.0.0 SaaS, 4B pulls, MP/KAYA, second GitHub bot user. "
    "Boundary only. No SaaS. No large pull. No second bot user."
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
    if SAAS_BIND:
        bad.append("saas_bind")
    if FOUR_B_PULL:
        bad.append("four_b_pull")
    if SECOND_BOT_USER:
        bad.append("second_bot_user")
    if BIND != "127.0.0.1":
        bad.append("bind")
    return {"ok": not bad, "reject": bad}


def advance_not_this_roadmap() -> dict:
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
        "saas_bind": SAAS_BIND,
        "four_b_pull": FOUR_B_PULL,
        "second_bot_user": SECOND_BOT_USER,
        "bind": BIND,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "Not this roadmap boundary: no 0.0.0.0 SaaS, no 4B pulls, no second GitHub bot user; opt-in AntiCon held",
    }


if __name__ == "__main__":
    out = advance_not_this_roadmap()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
