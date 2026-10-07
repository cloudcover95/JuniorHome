#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from deck_feed import feed

if __name__ == "__main__":
    row = feed()
    print(json.dumps({"n": row["n"], "energy": row["energy"], "adc": False}, indent=2))
