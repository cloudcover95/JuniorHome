#!/usr/bin/env python3
"""Prod runner for ROADMAP #4 Asahi MLX advance.
Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
Probe only from automations. Never imports mlx. Never boots UE.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from advance_asahi_mlx import advance_asahi_mlx

if __name__ == "__main__":
    out = advance_asahi_mlx()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
