#!/usr/bin/env python3
"""Advance the Order (do not skip) named item from docs/ROADMAP.md.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
No UE boot. Stdlib-lean. Does not wrap path/chain_record/record_block without naming the item.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from path import path

ITEM = (
    "ROADMAP Order (do not skip): 1. T13+ juniorctl skill-pin height "
    "(overnight owns). 2. GGUF operator sets JUNIOR_GGUF (bots do not download). "
    "3. i2sd userspace 127.0.0.1:8767 no docker.sock. 4. Asahi MLX probe only. "
    "5. UE5 launched JUNIOR_UE5=1 and watts>=80, default FrameForge2D, "
    "os_route.launch false until then."
)


def advance_order_do_not_skip_via_path() -> dict:
    # Hits name the item; path applies opt-in / AntiCon / one trit / base rate / held pool.
    hits = ["order", "do_not_skip", "T13+", "GGUF", "i2sd", "Asahi", "UE5"]
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
        "note": "order do not skip; sequence held via existing path; no downloads, no ue5 until gated",
    }


if __name__ == "__main__":
    out = advance_order_do_not_skip_via_path()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
