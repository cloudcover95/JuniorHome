#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from surface_compute import compute
from web3_receipt import receipt

if __name__ == "__main__":
    spun = compute("web", "web3")
    print(json.dumps(receipt("web3", spun.get("pack5", "")), indent=2))
