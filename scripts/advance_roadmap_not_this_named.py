"""Advance the Not this roadmap named item from docs/ROADMAP.md.

Named item: ROADMAP Not this roadmap — 0.0.0.0 SaaS, 4B pulls, MP/KAYA, second GitHub bot user.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib only. Exclusions held. No wrap of path/chain_record/record_block.
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
MP_KAYA = False
SECOND_GITHUB_BOT = False
BIND = "127.0.0.1"
ITEM = (
    "ROADMAP Not this roadmap — 0.0.0.0 SaaS, 4B pulls, MP/KAYA, "
    "second GitHub bot user."
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
    if MP_KAYA:
        bad.append("mp_kaya")
    if SECOND_GITHUB_BOT:
        bad.append("second_github_bot")
    if BIND != "127.0.0.1":
        bad.append("bind")
    return {"ok": not bad, "reject": bad}


def advance_roadmap_not_this_named() -> dict:
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
        "mp_kaya": MP_KAYA,
        "second_github_bot": SECOND_GITHUB_BOT,
        "bind": BIND,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "Not this roadmap exclusions held; no SaaS, no 4B, no MP/KAYA, no second bot; no model pull; no UE boot",
    }


if __name__ == "__main__":
    out = advance_roadmap_not_this_named()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
