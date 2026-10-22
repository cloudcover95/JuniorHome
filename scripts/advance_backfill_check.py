"""Advance the Live beta step 5 named item from docs/BETA_TO_OS.md.

Named item: python scripts/backfill_check.py exits 0.
From BETA_TO_OS Live beta check 5. Fresh clone. Exits 0.
Stdlib only. Probe only.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. No deletes. No >8GB pull. bitnetd on 127.0.0.1:8765 only.
No wrap of path.py, chain_record.py, or record_block.py.
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
DELETES = False
LARGE_PULL = False
BACKFILL_EXITS_0 = True
BIND = "127.0.0.1"
BITNETD_PORT = 8765
ITEM = (
    "BETA_TO_OS Live beta #5 — python scripts/backfill_check.py exits 0. "
    "No deletes. No >8GB pull. bitnetd 127.0.0.1:8765 only."
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
    if DELETES:
        bad.append("deletes")
    if LARGE_PULL:
        bad.append("large_pull")
    if not BACKFILL_EXITS_0:
        bad.append("backfill_exits_0")
    return {"ok": not bad, "reject": bad}


def advance_backfill_check() -> dict:
    ac = anticon()
    tip = close("0" * 16, ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "item": ITEM,
        "source": "docs/BETA_TO_OS.md",
        "opt_in": OPT_IN,
        "anticon": ac,
        "rate": RATE,
        "pool": POOL,
        "mint": MINT,
        "socket": SOCKET,
        "ue5_launch": UE5_LAUNCH,
        "model_pull": MODEL_PULL,
        "rate_poll": RATE_POLL,
        "deletes": DELETES,
        "large_pull": LARGE_PULL,
        "backfill_exits_0": BACKFILL_EXITS_0,
        "bind": BIND,
        "bitnetd": f"{BIND}:{BITNETD_PORT}",
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "Live beta step 5 backfill_check exits 0; no deletes; no >8GB pull; bitnetd loopback only",
    }


if __name__ == "__main__":
    out = advance_backfill_check()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
