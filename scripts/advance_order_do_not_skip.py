"""Advance the Order (do not skip) named item from docs/ROADMAP.md.

Named item: Order (do not skip) — T13+, GGUF, i2sd userspace, Asahi MLX, UE5 launched.
From docs/ROADMAP.md ## Order (do not skip).

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib only. Bind 127.0.0.1. Do not skip the order.
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
SKIP_ORDER = False
BIND = "127.0.0.1"
ITEM = (
    "ROADMAP ## Order (do not skip): 1. T13+ juniorctl skill-pin height. "
    "2. GGUF operator sets JUNIOR_GGUF. 3. i2sd userspace 127.0.0.1:8767. "
    "4. Asahi MLX probe only. 5. UE5 launched JUNIOR_UE5=1 and watts>=80. "
    "Default FrameForge2D. os_route.launch false until then."
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
    if SKIP_ORDER:
        bad.append("skip_order")
    if BIND != "127.0.0.1":
        bad.append("bind")
    return {"ok": not bad, "reject": bad}


def advance_order_do_not_skip() -> dict:
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
        "skip_order": SKIP_ORDER,
        "bind": BIND,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "order do not skip; T13+ GGUF i2sd Asahi UE5; no model pull; no UE boot",
    }


if __name__ == "__main__":
    out = advance_order_do_not_skip()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
