#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from wallet_class import bench

if __name__ == "__main__":
    print(json.dumps(bench(), indent=2))
