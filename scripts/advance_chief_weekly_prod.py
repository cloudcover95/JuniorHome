#!/usr/bin/env python3
"""Prod runner for ROADMAP JuniorCloud chief weekly advance.
Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
Only weekly. Writes docs/WEEKLY_AUDIT.md. Chief reads; does not build.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from advance_chief_weekly import advance_chief_weekly

if __name__ == "__main__":
    out = advance_chief_weekly()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
