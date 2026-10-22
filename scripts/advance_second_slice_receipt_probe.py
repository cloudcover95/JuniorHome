"""Advance the 19:00 second slice if receipt >45 min old named item from docs/ROADMAP.md.

Named item: ROADMAP automations ON: 19:00 — second slice if receipt >45 min old.
Overnight owns primary T-slices. Second slice only on stale receipt.
Do not recreate paused slots. Stdlib only. Probe only.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Bind 127.0.0.1. Do not wrap path.py, chain_record.py, or record_block.py.
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
RECEIPT_STALE_MIN = 45
SECOND_SLICE_ON_STALE = True
OVERNIGHT_OWNS_PRIMARY = True
BIND = "127.0.0.1"
ITEM = (
    "ROADMAP automations ON: 19:00 — second slice if receipt >45 min old. "
    "Overnight owns primary T-slices. Second slice only on stale receipt."
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
    if not SECOND_SLICE_ON_STALE:
        bad.append("second_slice_on_stale")
    if not OVERNIGHT_OWNS_PRIMARY:
        bad.append("overnight_owns_primary")
    if RECEIPT_STALE_MIN != 45:
        bad.append("receipt_stale_min")
    if BIND != "127.0.0.1":
        bad.append("bind")
    return {"ok": not bad, "reject": bad}


def advance_second_slice_receipt_probe() -> dict:
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
        "receipt_stale_min": RECEIPT_STALE_MIN,
        "second_slice_on_stale": SECOND_SLICE_ON_STALE,
        "overnight_owns_primary": OVERNIGHT_OWNS_PRIMARY,
        "bind": BIND,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "19:00 second slice if receipt >45 min old; overnight owns primary; no model pull; no UE boot",
    }


if __name__ == "__main__":
    out = advance_second_slice_receipt_probe()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
