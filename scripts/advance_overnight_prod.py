#!/usr/bin/env python3
"""Prod runner for ROADMAP 01:00 overnight T-slices advance.
Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
Overnight owns next T13 height. Bots do not advance height.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from advance_overnight import advance_overnight

if __name__ == "__main__":
    out = advance_overnight()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
