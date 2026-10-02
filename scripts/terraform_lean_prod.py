#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from terraform_lean import run

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
