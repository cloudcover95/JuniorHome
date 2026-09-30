#!/usr/bin/env python3
"""Local Flagstaff AND gate for deck addons. No 0.0.0.0."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from deck_port import gate, log_note

votes = [True, True, True, True, True, True]
print(json.dumps({"flagstaff": gate(votes), "row": log_note("osai gate")}, indent=2))
