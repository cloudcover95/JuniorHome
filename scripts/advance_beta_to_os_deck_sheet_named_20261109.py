#!/usr/bin/env python3
"""Advance the Deck sheet named item from docs/BETA_TO_OS.md.

Named item: Deck sheet — Deck controls and Gaia helper stay separate.
Package ports: JuniorLLM, AGI_SDK, JuniorOSai, JuniorOS, web3node, obsidian.
Vault not synced. Device not opened. Schematic: JuniorHome docs/deck_capability.svg.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib only. Does not wrap path.py, chain_record.py, or record_block.py.
Additive only. Cap 1 file.
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
MODEL_PULL = False
RATE_POLL = False
UE5_LAUNCH = False
DECK_CONTROLS_SEPARATE = True
GAIA_HELPER_SEPARATE = True
VAULT_NOT_SYNCED = True
DEVICE_NOT_OPENED = True
ITEM = (
    "BETA_TO_OS Deck sheet — Deck controls and Gaia helper stay separate. "
    "Package ports: JuniorLLM, AGI_SDK, JuniorOSai, JuniorOS, web3node, obsidian. "
    "Vault not synced. Device not opened. Schematic: JuniorHome docs/deck_capability.svg."
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
    if MODEL_PULL:
        bad.append("model_pull")
    if RATE_POLL:
        bad.append("rate_poll")
    if UE5_LAUNCH:
        bad.append("ue5_launch")
    if not DECK_CONTROLS_SEPARATE:
        bad.append("deck_controls_separate")
    if not GAIA_HELPER_SEPARATE:
        bad.append("gaia_helper_separate")
    if not VAULT_NOT_SYNCED:
        bad.append("vault_not_synced")
    if not DEVICE_NOT_OPENED:
        bad.append("device_not_opened")
    return {"ok": not bad, "reject": bad}


def advance() -> dict:
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
        "model_pull": MODEL_PULL,
        "rate_poll": RATE_POLL,
        "ue5_launch": UE5_LAUNCH,
        "deck_controls_separate": DECK_CONTROLS_SEPARATE,
        "gaia_helper_separate": GAIA_HELPER_SEPARATE,
        "vault_not_synced": VAULT_NOT_SYNCED,
        "device_not_opened": DEVICE_NOT_OPENED,
        "hash": tip,
        "trit_pack": list(TRIT_PACK),
        "advanced": ac["ok"],
        "note": (
            "Deck sheet: controls and Gaia helper separate; package ports held; "
            "vault not synced; device not opened; path held at rate 0.05; "
            "opt-in AntiCon one trit pack; no model pull; no rate poll; "
            "pool held; mint false; socket false; no UE boot"
        ),
    }


if __name__ == "__main__":
    out = advance()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
