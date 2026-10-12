#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from juniors_genesis import genesis

if __name__ == "__main__":
    row = genesis()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
