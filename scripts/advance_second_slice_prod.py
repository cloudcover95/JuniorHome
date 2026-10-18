#!/usr/bin/env python3
"""Prod runner for ROADMAP automations 19:00 second slice advance.
Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
Second slice if receipt >45 min old. Bots do not recreate paused slots.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from advance_second_slice import advance_second_slice

if __name__ == "__main__":
    out = advance_second_slice()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
