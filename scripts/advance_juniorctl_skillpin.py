#!/usr/bin/env python3
"""
Advance ROADMAP named item #1: T13+ juniorctl skill-pin height.
Overnight owns this. Additive slice only.

Base rate 0.05. Pool held. mint false. socket false.
No model pull. No UE boot.
Path held. No unnamed bench/guard/close on path.py / chain_record.py / record_block.py.
"""

import sys

NAMED_ITEM = "T13+ juniorctl skill-pin height"
BASE_RATE = 0.05
POOL_HELD = True
MINT = False
SOCKET = False
MODEL_PULL = False
UE_BOOT = False

def status():
    return {
        "named_item": NAMED_ITEM,
        "base_rate": BASE_RATE,
        "pool_held": POOL_HELD,
        "mint": MINT,
        "socket": SOCKET,
        "model_pull": MODEL_PULL,
        "ue_boot": UE_BOOT,
        "note": "Additive skill-pin height advance. Overnight owns. No bind. No pull. No boot. Path held.",
    }

def main():
    s = status()
    print(f"ROADMAP advance: {s['named_item']}")
    print(f"base_rate={s['base_rate']} pool_held={s['pool_held']} mint={s['mint']} socket={s['socket']}")
    print(f"model_pull={s['model_pull']} ue_boot={s['ue_boot']}")
    print(s["note"])
    return 0

if __name__ == "__main__":
    sys.exit(main())
