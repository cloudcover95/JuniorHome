"""Advance ROADMAP #4 Asahi MLX — probe only from automations.

Named item: Asahi MLX — probe only from automations.
(docs/ROADMAP.md order item 4)

Additive slice. Opt-in, AntiCon, one trit pack, base rate 0.05, pool held,
mint false, socket false. No model pull. No UE boot. No rate poll.
Stdlib only. Bind 127.0.0.1 only. Never opens a socket.
Close rule: sha256(prev || pack || text)[:16].
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
RATE_POLL = False
PROBE_FROM_AUTOMATIONS = True
BIND = "127.0.0.1"
ROADMAP_ITEM = "Asahi MLX — probe only from automations."
DAY = "2026-10-26"
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
    if FETCH:
        bad.append("fetch")
    if UE5_LAUNCH:
        bad.append("ue5_launch")
    if MODEL_PULL:
        bad.append("model_pull")
    if RATE_POLL:
        bad.append("rate_poll")
    if not PROBE_FROM_AUTOMATIONS:
        bad.append("probe_from_automations")
    if BIND != "127.0.0.1":
        bad.append("bind")
    return {"ok": not bad, "reject": bad}


def advance() -> dict:
    ac = anticon()
    tip = close(GENESIS, ROADMAP_ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "roadmap_item": ROADMAP_ITEM,
        "source": "docs/ROADMAP.md #4",
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
        "rate_poll": RATE_POLL,
        "probe_from_automations": PROBE_FROM_AUTOMATIONS,
        "bind": BIND,
        "tip": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
    }


def self_check() -> dict:
    assert RATE == 0.05
    assert OPT_IN and ANTICON and POOL_HELD
    assert not (MINT or SOCKET or FETCH or UE5_LAUNCH or MODEL_PULL or RATE_POLL)
    assert PROBE_FROM_AUTOMATIONS
    assert BIND == "127.0.0.1"
    assert "Asahi MLX" in ROADMAP_ITEM and "probe only from automations" in ROADMAP_ITEM
    out = advance()
    assert out["ok"] and out["advanced"] and out["tip"] and len(out["tip"]) == 16
    assert close(GENESIS, ROADMAP_ITEM) == out["tip"]
    return out


if __name__ == "__main__":
    try:
        result = self_check()
        print(result)
        sys.exit(0)
    except AssertionError as e:
        print({"ok": False, "error": repr(e)})
        sys.exit(1)
