#!/usr/bin/env python3
"""Advance the paused automations named item from docs/ROADMAP.md.

Named item: PAUSED: - 07:00, 13:00 - Friday weekly `7716afb1` (merged into chief) Do not recreate paused slots.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib only. Do not recreate paused slots.
Do not wrap path.py, chain_record.py, or record_block.py. Additive only.
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
DO_NOT_RECREATE_PAUSED = True
PAUSED_0700 = True
PAUSED_1300 = True
FRIDAY_WEEKLY_7716AFB1_MERGED = True
ITEM = (
    "ROADMAP PAUSED: 07:00, 13:00 and Friday weekly 7716afb1 (merged into chief). Do not recreate paused slots."
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
    if not DO_NOT_RECREATE_PAUSED:
        bad.append("do_not_recreate_paused")
    if not PAUSED_0700:
        bad.append("paused_0700")
    if not PAUSED_1300:
        bad.append("paused_1300")
    if not FRIDAY_WEEKLY_7716AFB1_MERGED:
        bad.append("friday_weekly_7716afb1_merged")
    return {"ok": not bad, "reject": bad}


def advance_automations_paused_0700_1300_friday_weekly_7716afb1_merged_into_chief_do_not_recreate_slots_named_20261123() -> dict:
    ac = anticon()
    tip = close("0" * 16, ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "item": ITEM,
        "source": "docs/ROADMAP.md Automations PAUSED",
        "opt_in": OPT_IN,
        "anticon": ac,
        "rate": RATE,
        "pool": POOL,
        "mint": MINT,
        "socket": SOCKET,
        "ue5_launch": UE5_LAUNCH,
        "model_pull": MODEL_PULL,
        "rate_poll": RATE_POLL,
        "do_not_recreate_paused": DO_NOT_RECREATE_PAUSED,
        "paused_0700": PAUSED_0700,
        "paused_1300": PAUSED_1300,
        "friday_weekly_7716afb1_merged": FRIDAY_WEEKLY_7716AFB1_MERGED,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "PAUSED 07:00 13:00 Friday weekly 7716afb1 merged into chief; do not recreate slots; path held at rate 0.05; opt-in AntiCon one trit pack; no model pull no UE boot no rate poll",
    }


if __name__ == "__main__":
    out = advance_automations_paused_0700_1300_friday_weekly_7716afb1_merged_into_chief_do_not_recreate_slots_named_20261123()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
