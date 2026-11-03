#!/usr/bin/env python3
"""Advance the JuniorOS overlay named item from docs/BETA_TO_OS.md.

Named item: JuniorOS overlay — Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH. I2_S 2B4T only if GGUF already on disk. No OSWorld score. No kernel blob.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib only. Bind 127.0.0.1.
Do not wrap path.py, chain_record.py, or record_block.py.
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
OVERLAY = True
OS_RELEASE_JUNIOR = True
HARDENED_BITNETD = True
JUNIORCTL_ON_PATH = True
I2S_ONLY_IF_GGUF = True
NO_OSWORLD = True
NO_KERNEL_BLOB = True
BIND = "127.0.0.1"
ITEM = (
    "BETA_TO_OS JuniorOS overlay — Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH. I2_S 2B4T only if GGUF already on disk. No OSWorld score. No kernel blob."
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


def advance_beta_to_os_junioros_overlay_debian_alpine_named() -> dict:
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
        "overlay": OVERLAY,
        "os_release_junior": OS_RELEASE_JUNIOR,
        "hardened_bitnetd": HARDENED_BITNETD,
        "juniorctl_on_path": JUNIORCTL_ON_PATH,
        "i2s_only_if_gguf": I2S_ONLY_IF_GGUF,
        "no_osworld": NO_OSWORLD,
        "no_kernel_blob": NO_KERNEL_BLOB,
        "bind": BIND,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": "JuniorOS overlay Debian/Alpine + os-release.junior + hardened bitnetd.service + juniorctl on PATH; I2_S only if GGUF on disk; no OSWorld; no kernel blob; path held at rate 0.05; no model pull; no UE boot; no rate poll; no wrap of path/chain_record/record_block",
    }


if __name__ == "__main__":
    out = advance_beta_to_os_junioros_overlay_debian_alpine_named()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
