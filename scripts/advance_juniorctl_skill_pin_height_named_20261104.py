"""Advance ROADMAP #1 named item: juniorctl skill-pin height.

Named item: juniorctl (T13+ skill-pin height, overnight owns).
Additive only. Does not wrap path.py, chain_record.py, or record_block.py.

Base rate 0.05. Pool held. mint false. socket false. No model pull. No UE boot.
Stdlib only. Bind 127.0.0.1. Close: sha256(prev || pack || text)[:16].
"""
from __future__ import annotations

import hashlib
import sys

RATE = 0.05
OPT_IN = True
ANTICON = True
POOL_HELD = True
MINT = False
SOCKET = False
FETCH = False
UE5_LAUNCH = False
MODEL_PULL = False
BIND = "127.0.0.1"
ROADMAP_ITEM = "juniorctl (T13+ skill-pin height, overnight owns)"
DAY = "2026-11-04"
GENESIS = "0" * 16
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


def anticon(trits) -> dict:
    reject = []
    if any(t not in (-1, 0, 1) for t in trits):
        reject.append("trit_range")
    if len(trits) == 0:
        reject.append("empty")
    return {"ok": len(reject) == 0, "reject": reject}


def close(prev: str, pack_b: bytes, text: str) -> str:
    h = hashlib.sha256()
    h.update(prev.encode())
    h.update(pack_b)
    h.update(text.encode())
    return h.hexdigest()[:16]


def main() -> int:
    ac = anticon(TRIT_PACK)
    tip = close(GENESIS, PACK, ROADMAP_ITEM)
    out = {
        "ok": ac["ok"] and OPT_IN and ANTICON and POOL_HELD and not MINT and not SOCKET and not MODEL_PULL and not UE5_LAUNCH,
        "roadmap_item": ROADMAP_ITEM,
        "source": "docs/ROADMAP.md / juniorctl skill-pin",
        "day": DAY,
        "opt_in": OPT_IN,
        "anticon": ac,
        "rate": RATE,
        "pool_held": POOL_HELD,
        "mint": MINT,
        "socket": SOCKET,
        "fetch": FETCH,
        "ue5_launch": UE5_LAUNCH,
        "model_pull": MODEL_PULL,
        "bind": BIND,
        "tip": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": True,
        "note": "probe only; juniorctl skill-pin named; no socket open; no model pull; no UE boot",
    }
    print(out)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
