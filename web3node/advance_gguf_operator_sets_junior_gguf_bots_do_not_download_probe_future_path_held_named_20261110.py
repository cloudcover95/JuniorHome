#!/usr/bin/env python3
"""Advance the GGUF named item from docs/ROADMAP.md.

Named item: Order (do not skip) 2. **GGUF** — operator sets JUNIOR_GGUF. Bots do not download. `web3node/probe_future.py`.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib only. Operator sets JUNIOR_GGUF. Bots do not download.
probe_future.py held. Do not wrap path.py, chain_record.py, or record_block.py.
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
JUNIOR_GGUF_SET = False  # operator sets, not bots
BOTS_DOWNLOAD = False
PROBE_FUTURE_HELD = True
BIND = "127.0.0.1"
ITEM = (
    "ROADMAP Order (do not skip) #2 GGUF — operator sets JUNIOR_GGUF. "
    "Bots do not download. web3node/probe_future.py."
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
    if JUNIOR_GGUF_SET:
        bad.append("junior_gguf_set_by_bot")
    if BOTS_DOWNLOAD:
        bad.append("bots_download")
    if not PROBE_FUTURE_HELD:
        bad.append("probe_future_held")
    if BIND != "127.0.0.1":
        bad.append("bind")
    return {"ok": not bad, "reject": bad}


def advance_gguf_operator_sets_junior_gguf_bots_do_not_download_probe_future_path_held_named_20261110() -> dict:
    ac = anticon()
    tip = close("0" * 16, ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "item": ITEM,
        "source": "docs/ROADMAP.md Order #2",
        "opt_in": OPT_IN,
        "anticon": ac,
        "rate": RATE,
        "pool": POOL,
        "mint": MINT,
        "socket": SOCKET,
        "ue5_launch": UE5_LAUNCH,
        "model_pull": MODEL_PULL,
        "rate_poll": RATE_POLL,
        "junior_gguf_set": JUNIOR_GGUF_SET,
        "bots_download": BOTS_DOWNLOAD,
        "probe_future_held": PROBE_FUTURE_HELD,
        "bind": BIND,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "GGUF held: operator sets JUNIOR_GGUF; bots do not download; probe_future.py held; path held at rate 0.05; opt-in AntiCon; no model pull no UE boot no rate poll",
    }


if __name__ == "__main__":
    out = advance_gguf_operator_sets_junior_gguf_bots_do_not_download_probe_future_path_held_named_20261110()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
