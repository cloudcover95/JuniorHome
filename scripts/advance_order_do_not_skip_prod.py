#!/usr/bin/env python3
"""Prod runner for ROADMAP Order (do not skip) named item.
Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
Confirms the sequence. Do not skip.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from advance_order_do_not_skip import advance_order_do_not_skip

if __name__ == "__main__":
    out = advance_order_do_not_skip()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
