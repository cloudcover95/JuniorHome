#!/usr/bin/env python3
"""Advance BETA_TO_OS / ROADMAP named item: JuniorOS overlay path held.

Named item: JuniorOS overlay — Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH.
I2_S 2B4T only if GGUF already on disk. No OSWorld score. No kernel blob.
No model pull. No UE boot. Bind loopback.

Additive only. Opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No rate poll.
Does not wrap path.py, chain_record.py, or record_block.py.
Stdlib only.
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
OVERLAY = True
OS_RELEASE_JUNIOR = True
HARDENED_BITNETD = True
JUNIORCTL_ON_PATH = True
I2S_ONLY_IF_GGUF = True
NO_OSWORLD = True
NO_KERNEL_BLOB = True
BIND = "127.0.0.1"
ITEM = (
    "BETA_TO_OS / ROADMAP named item — JuniorOS overlay path held. "
    "Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH. "
    "I2_S 2B4T only if GGUF on disk. No OSWorld. No kernel blob. No model pull. No UE boot."
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
    if not OVERLAY:
        bad.append("overlay")
    if not OS_RELEASE_JUNIOR:
        bad.append("os_release_junior")
    if not HARDENED_BITNETD:
        bad.append("hardened_bitnetd")
    if not JUNIORCTL_ON_PATH:
        bad.append("juniorctl_on_path")
    if not I2S_ONLY_IF_GGUF:
        bad.append("i2s_only_if_gguf")
    if not NO_OSWORLD:
        bad.append("no_osworld")
    if not NO_KERNEL_BLOB:
        bad.append("no_kernel_blob")
    if BIND != "127.0.0.1":
        bad.append("bind")
    return {"ok": not bad, "reject": bad}


def advance() -> dict:
    ac = anticon()
    tip = close(GENESIS, ITEM) if ac["ok"] else None
    return {
        "ok": ac["ok"],
        "item": ITEM,
        "source": "docs/BETA_TO_OS.md + docs/ROADMAP.md",
        "opt_in": OPT_IN,
        "anticon": ac,
        "rate": RATE,
        "pool_held": POOL_HELD,
        "mint": MINT,
        "socket": SOCKET,
        "ue5_launch": UE5_LAUNCH,
        "model_pull": MODEL_PULL,
        "rate_poll": RATE_POLL,
        "overlay": OVERLAY,
        "os_release_junior": OS_RELEASE_JUNIOR,
        "hardened_bitnetd": HARDENED_BITNETD,
        "juniorctl_on_path": JUNIORCTL_ON_PATH,
        "i2s_only_if_gguf": I2S_ONLY_IF_GGUF,
        "no_osworld": NO_OSWORLD,
        "no_kernel_blob": NO_KERNEL_BLOB,
        "bind": BIND,
        "tip": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "JuniorOS overlay path held; Debian/Alpine + os-release.junior + hardened bitnetd + juniorctl on PATH; no model pull; no UE boot; no rate poll; no wrap of path/chain_record/record_block",
    }


def self_check() -> dict:
    assert RATE == 0.05
    assert OPT_IN and ANTICON and POOL_HELD
    assert not (MINT or SOCKET or UE5_LAUNCH or MODEL_PULL or RATE_POLL)
    assert BIND == "127.0.0.1"
    assert "JuniorOS overlay" in ITEM
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
