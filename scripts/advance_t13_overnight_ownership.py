"""Advance the T13+ named item from docs/ROADMAP.md.

Named item: T13+ — juniorctl skill-pin height. Overnight owns this.
From ROADMAP #1. Order do not skip.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib only. Overnight owns skill-pin height. Bots do not advance height.
No wrap of path/chain_record/record_block.
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
JUNIORCTL_IMPORT = False
HEIGHT_ADVANCE = False
OVERNIGHT_OWNS = True
ITEM = (
    "ROADMAP #1 T13+ — juniorctl skill-pin height. Overnight owns this. "
    "Bots do not advance height."
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
    if JUNIORCTL_IMPORT:
        bad.append("juniorctl_import")
    if HEIGHT_ADVANCE:
        bad.append("height_advance")
    if not OVERNIGHT_OWNS:
        bad.append("overnight_owns")
    return {"ok": not bad, "reject": bad}


def advance_t13_overnight_ownership() -> dict:
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
        "juniorctl_import": JUNIORCTL_IMPORT,
        "height_advance": HEIGHT_ADVANCE,
        "overnight_owns": OVERNIGHT_OWNS,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "overnight owns skill-pin height; bots do not advance; no model pull; no UE boot",
    }


if __name__ == "__main__":
    out = advance_t13_overnight_ownership()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
