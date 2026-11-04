#!/usr/bin/env python3
"""Advance BETA_TO_OS Live beta named item: fresh clone checks.

Named item: Live beta — A fresh clone:
1. PYTHONPATH=. python rails/linux/juniorctl.py health
2. juniorctl ask / night / security
3. python tests/test_junior_aie.py and tests/test_linux_security.py
4. JuniorClimbs scripts/stonefield_stdlib_probe.py (no FastAPI required)
5. python scripts/backfill_check.py exits 0
6. No deletes. No >8GB pull. bitnetd on 127.0.0.1:8765 only
(docs/BETA_TO_OS.md in JuniorLLM, referenced from JuniorHome).

Additive only. Opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate poll.
No UE boot. Stdlib only. Bind 127.0.0.1.
Does not wrap path.py, chain_record.py, or record_block.py.
"""
from __future__ import annotations

import hashlib
import json
import sys

RATE = 0.05
OPT_IN = True
ANTICON = True
POOL_HELD = True
MINT = False
SOCKET = False
UE5_LAUNCH = False
MODEL_PULL = False
RATE_POLL = False
DELETES = False
LARGE_PULL = False
BITNETD_LOOPBACK = True
BIND = "127.0.0.1"
BITNETD_PORT = 8765
ITEM = (
    "BETA_TO_OS Live beta — A fresh clone: "
    "1. PYTHONPATH=. python rails/linux/juniorctl.py health "
    "2. juniorctl ask / night / security "
    "3. python tests/test_junior_aie.py and tests/test_linux_security.py "
    "4. JuniorClimbs scripts/stonefield_stdlib_probe.py (no FastAPI required) "
    "5. python scripts/backfill_check.py exits 0 "
    "6. No deletes. No >8GB pull. bitnetd on 127.0.0.1:8765 only."
)
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


def close(prev: str, text: str) -> str:
    return hashlib.sha256(prev.encode() + PACK + text.encode()).hexdigest()[:16]


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
    if not BITNETD_LOOPBACK:
        bad.append("bitnetd_loopback")
    if BIND != "127.0.0.1":
        bad.append("bind")
    return {"ok": not bad, "reject": bad}


def advance() -> dict:
    ac = anticon()
    tip = close(GENESIS, ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "item": ITEM,
        "source": "docs/BETA_TO_OS.md Live beta",
        "opt_in": OPT_IN,
        "anticon": ac,
        "rate": RATE,
        "pool_held": POOL_HELD,
        "mint": MINT,
        "socket": SOCKET,
        "ue5_launch": UE5_LAUNCH,
        "model_pull": MODEL_PULL,
        "rate_poll": RATE_POLL,
        "deletes": DELETES,
        "large_pull": LARGE_PULL,
        "bitnetd_loopback": BITNETD_LOOPBACK,
        "bind": BIND,
        "bitnetd": f"{BIND}:{BITNETD_PORT}",
        "tip": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "Live beta fresh clone checks named; no deletes no large pull bitnetd loopback; path held; no model pull; no UE boot; no rate poll; no wrap of path/chain_record/record_block",
    }


def self_check() -> dict:
    assert RATE == 0.05
    assert OPT_IN and ANTICON and POOL_HELD
    assert not (MINT or SOCKET or UE5_LAUNCH or MODEL_PULL or RATE_POLL or DELETES or LARGE_PULL)
    assert BITNETD_LOOPBACK
    assert BIND == "127.0.0.1"
    assert "Live beta" in ITEM
    assert "fresh clone" in ITEM
    assert "bitnetd on 127.0.0.1:8765 only" in ITEM
    out = advance()
    assert out["ok"] and out["advanced"] and out["tip"] and len(out["tip"]) == 16
    assert close(GENESIS, ITEM) == out["tip"]
    return out


if __name__ == "__main__":
    try:
        result = self_check()
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except AssertionError as e:
        print(json.dumps({"ok": False, "error": repr(e)}))
        sys.exit(1)
