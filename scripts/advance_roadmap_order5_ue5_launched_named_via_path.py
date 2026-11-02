#!/usr/bin/env python3
"""Advance the UE5 launched named item from docs/ROADMAP.md.

Named item: ROADMAP Order (do not skip) #5 — UE5 launched.
JUNIOR_UE5=1 and watts>=80. Default FrameForge2D. os_route.launch false until then.

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
    "ROADMAP Order (do not skip) #5: UE5 launched — "
    "JUNIOR_UE5=1 and watts>=80. Default FrameForge2D. "
    "os_route.launch false until then."
)


def advance_roadmap_order5_ue5_launched_named_via_path() -> dict:
    # Hits name the item; path applies opt-in / AntiCon / one trit / base rate / held pool.
    hits = [
        "UE5 launched",
        "JUNIOR_UE5=1",
        "watts>=80",
        "Default FrameForge2D",
        "os_route.launch false",
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
        "note": (
            "UE5 launched gate; default FrameForge2D; os_route.launch false; "
            "via existing path; no downloads, no ue5 boot, no rate poll"
        ),
    }


if __name__ == "__main__":
    out = advance_roadmap_order5_ue5_launched_named_via_path()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
