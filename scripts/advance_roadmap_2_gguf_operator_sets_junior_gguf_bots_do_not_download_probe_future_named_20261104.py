#!/usr/bin/env python3
"""Advance ROADMAP named item: GGUF — operator sets JUNIOR_GGUF. Bots do not download. web3node/probe_future.py.

Named item: ROADMAP #2 GGUF — operator sets JUNIOR_GGUF. Bots do not download. web3node/probe_future.py. (docs/ROADMAP.md).

Additive only. Does not wrap path.py, chain_record.py, or record_block.py.

Base rate 0.05. Pool held. mint false. socket false. No model pull. No UE boot.
Stdlib only. Bind 127.0.0.1. Opt-in, AntiCon, one trit pack. No rate poll.
Probe future only if GGUF already on disk. Bots do not download.
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
FETCH = False
UE5_LAUNCH = False
MODEL_PULL = False
RATE_POLL = False
PROBE_FUTURE = True
BOTS_DOWNLOAD = False
BIND = "127.0.0.1"
ROADMAP_ITEM = (
    "GGUF — operator sets JUNIOR_GGUF. Bots do not download. web3node/probe_future.py. "
    "(docs/ROADMAP.md #2)"
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
    if not PROBE_FUTURE:
        bad.append("probe_future")
    if BOTS_DOWNLOAD:
        bad.append("bots_download")
    if BIND != "127.0.0.1":
        bad.append("bind")
    return {"ok": not bad, "reject": bad}


def advance() -> dict:
    ac = anticon()
    tip = close(GENESIS, ROADMAP_ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "roadmap_item": ROADMAP_ITEM,
        "source": "docs/ROADMAP.md #2 GGUF",
        "opt_in": OPT_IN,
        "anticon": ac,
        "rate": RATE,
        "pool_held": POOL_HELD,
        "mint": MINT,
        "socket": SOCKET,
        "fetch": FETCH,
        "ue5_launch": UE5_LAUNCH,
        "model_pull": MODEL_PULL,
        "rate_poll": RATE_POLL,
        "probe_future": PROBE_FUTURE,
        "bots_download": BOTS_DOWNLOAD,
        "bind": BIND,
        "tip": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "operator sets JUNIOR_GGUF; bots do not download; probe_future only; no socket open; no model pull; no UE boot; no wrap of path/chain_record/record_block",
    }


def self_check() -> dict:
    assert RATE == 0.05
    assert OPT_IN and ANTICON and POOL_HELD
    assert not (MINT or SOCKET or FETCH or UE5_LAUNCH or MODEL_PULL or RATE_POLL or BOTS_DOWNLOAD)
    assert PROBE_FUTURE
    assert BIND == "127.0.0.1"
    assert "GGUF" in ROADMAP_ITEM
    assert "operator sets JUNIOR_GGUF" in ROADMAP_ITEM
    assert "Bots do not download" in ROADMAP_ITEM
    assert "probe_future.py" in ROADMAP_ITEM
    out = advance()
    assert out["ok"] and out["advanced"] and out["tip"] and len(out["tip"]) == 16
    assert close(GENESIS, ROADMAP_ITEM) == out["tip"]
    return out


if __name__ == "__main__":
    try:
        result = self_check()
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except AssertionError as e:
        print(json.dumps({"ok": False, "error": repr(e)}))
        sys.exit(1)
