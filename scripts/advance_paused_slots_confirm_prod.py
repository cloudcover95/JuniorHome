#!/usr/bin/env python3
"""Prod runner for ROADMAP: Do not recreate paused slots.

PAUSED: 07:00, 13:00, Friday weekly 7716afb1 (merged into chief).
Do not recreate paused slots.

Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from advance_paused_slots_confirm import advance_paused_slots_confirm

if __name__ == "__main__":
    out = advance_paused_slots_confirm()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
