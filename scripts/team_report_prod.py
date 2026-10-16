#!/usr/bin/env python3
"""Team report one-shot. Local only."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from web3node.team_report import report

if __name__ == "__main__":
    team = sys.argv[1] if len(sys.argv) > 1 else "llc"
    print(json.dumps(report(team), indent=2, default=str))
