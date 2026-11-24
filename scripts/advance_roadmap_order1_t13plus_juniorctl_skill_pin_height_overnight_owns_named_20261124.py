#!/usr/bin/env python3
"""Advance the T13+ named item from docs/ROADMAP.md.

Named item: Order (do not skip) #1 — T13+ — juniorctl skill-pin height. Overnight owns this.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib-lean. Hits name the item. No wrap of path.py that names no item.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from path import path

ITEM = (
    "ROADMAP Order (do not skip) #1: T13+ — juniorctl skill-pin height. "
    "Overnight owns this."
)


def advance_roadmap_order1_t13plus_juniorctl_skill_pin_height_overnight_owns_named_20261124() -> dict:
    # Hits name the item; path applies opt-in / AntiCon / one trit / base rate / held pool.
    hits = [
        "T13+",
        "juniorctl",
        "skill-pin",
        "height",
        "Overnight",
        "owns",
    ]
    row = path("roadmap", hits, True)
    return {
        "ok": row.get("ok", False),
        "item": ITEM,
        "source": "docs/ROADMAP.md",
        "path": row,
        "opt_in": True,
        "rate": row.get("rate", 0.05),
        "pool": "held",
        "mint": False,
        "socket": False,
        "advanced": row.get("ok", False),
        "note": "T13+ juniorctl skill-pin height; overnight owns; via existing path; no model pull, no ue5, no rate poll",
    }


if __name__ == "__main__":
    out = advance_roadmap_order1_t13plus_juniorctl_skill_pin_height_overnight_owns_named_20261124()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
