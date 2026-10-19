#!/usr/bin/env python3
"""Prod runner for ROADMAP after 2026-09-13 cap advance.
Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from advance_roadmap_after_cap import advance_roadmap_after_cap

if __name__ == "__main__":
    out = advance_roadmap_after_cap()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
