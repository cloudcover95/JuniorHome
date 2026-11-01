#!/usr/bin/env python3
"""Advance the i2sd userspace named item from docs/ROADMAP.md.

Named item: Order (do not skip) 3. **i2sd userspace** — 127.0.0.1:8767. No docker.sock.

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
    "ROADMAP Order (do not skip) #3: i2sd userspace — 127.0.0.1:8767. No docker.sock."
)


def advance_i2sd_userspace_via_path() -> dict:
    # Hits name the item; path applies opt-in / AntiCon / one trit / base rate / held pool.
    hits = ["i2sd", "userspace", "127.0.0.1:8767", "no-docker.sock"]
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
        "note": "i2sd userspace 127.0.0.1:8767 no docker.sock; via existing path; no model pull, no ue5, no rate poll",
    }


if __name__ == "__main__":
    out = advance_i2sd_userspace_via_path()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
