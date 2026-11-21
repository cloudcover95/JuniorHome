#!/usr/bin/env python3
"""Advance ROADMAP #1: T13+ juniorctl skill-pin height.

Named item: juniorctl (ROADMAP order 1 — T13+ skill-pin height).
Additive only. Does not wrap path.py, chain_record.py, or record_block.py.

Base rate 0.05. Pool held. mint false. socket false.
No model pull. No UE boot. Stdlib only. Bind 127.0.0.1.
"""
from __future__ import annotations

RATE = 0.05
POOL_HELD = True
MINT = False
SOCKET = False
MODEL_PULL = False
UE_BOOT = False
BIND = "127.0.0.1"
ROADMAP_ITEM = "juniorctl skill-pin height (ROADMAP #1 T13+)"
DAY = "2026-11-21"
HEIGHT = 13

def status() -> dict:
    return {
        "item": ROADMAP_ITEM,
        "height": HEIGHT,
        "base_rate": RATE,
        "pool_held": POOL_HELD,
        "mint": MINT,
        "socket": SOCKET,
        "model_pull": MODEL_PULL,
        "ue_boot": UE_BOOT,
        "bind": BIND,
        "day": DAY,
        "live": False,
    }

if __name__ == "__main__":
    print(status())
