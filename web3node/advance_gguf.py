"""Advance the GGUF named item from docs/ROADMAP.md.
Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
Operator sets JUNIOR_GGUF. Bots do not download.
"""
import json
import sys

from path import path
from probe_future import probe


def advance_gguf():
    p = probe()
    hits = ["gguf", "roadmap"]
    if p.get("gguf_exists"):
        hits.append("present")
    row = path("operator", hits, True)
    return {
        "item": "GGUF",
        "source": "docs/ROADMAP.md",
        "probe": p,
        "path": row,
        "advanced": row.get("ok", False),
        "note": "operator sets JUNIOR_GGUF; bots do not download",
    }


if __name__ == "__main__":
    out = advance_gguf()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["advanced"] else 1)
