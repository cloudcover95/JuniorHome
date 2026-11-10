#!/usr/bin/env python3
"""Advance the 01:00 overnight T-slices named item from docs/ROADMAP.md.

Named item: ROADMAP Automations (current) ON: 01:00 overnight — T-slices (next T13 height).

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib-lean. Hits name the item.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from path import path

ITEM = (
    "ROADMAP Automations (current) ON: 01:00 overnight — T-slices (next T13 height)."
)


def advance_roadmap_0100_overnight_t_slices_next_t13_height_named_via_path_20261110() -> dict:
    # Hits name the item; path applies opt-in / AntiCon / one trit pack / base rate / held pool.
    hits = ["01:00 overnight", "T-slices", "next T13 height"]
    row = path("roadmap", hits, True)
    advanced = (
        row.get("ok") is True
        and row.get("rate") == 0.05
        and row.get("mint") is False
        and row.get("socket") is False
    )
    return {
        "ok": advanced,
        "item": ITEM,
        "source": "docs/ROADMAP.md",
        "path": row,
        "opt_in": True,
        "rate": row.get("rate"),
        "pool": "held",
        "mint": False,
        "socket": False,
        "advanced": advanced,
        "note": (
            "01:00 overnight T-slices next T13 height named; via existing path; "
            "opt-in AntiCon one trit pack base rate 0.05 pool held mint false socket false; "
            "no downloads no ue5 boot no rate poll"
        ),
    }


if __name__ == "__main__":
    out = advance_roadmap_0100_overnight_t_slices_next_t13_height_named_via_path_20261110()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
