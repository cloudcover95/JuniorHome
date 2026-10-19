#!/usr/bin/env python3
"""Prod runner for ROADMAP JuniorCloud chief weekly Sun/Mon 17:00 MDT advance.
Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
Only weekly. dd0b1c93. Writes docs/WEEKLY_AUDIT.md. Do not boot UE.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from advance_chief_weekly_schedule import advance_chief_weekly_schedule

if __name__ == "__main__":
    out = advance_chief_weekly_schedule()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
