#!/usr/bin/env python3
"""Prod runner for ROADMAP #5 UE5 launched advance.
Uses the existing path: opt-in, AntiCon, one trit pack, base rate 0.05,
pool held, mint false, socket false. No model pull. No rate set by poll.
JUNIOR_UE5=1 and watts>=80 for eligibility. Default FrameForge2D.
os_route.launch false. Bots do not launch UE5. Do not boot UE.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from advance_ue5 import advance_ue5

if __name__ == "__main__":
    out = advance_ue5()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("advanced") else 1)
