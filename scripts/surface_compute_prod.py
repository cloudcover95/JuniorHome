#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from surface_compute import compute

if __name__ == "__main__":
    print(json.dumps(compute(sys.argv[1] if len(sys.argv) > 1 else "web"), indent=2))
