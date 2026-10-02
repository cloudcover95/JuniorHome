#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from suite_intervene import intervene

if __name__ == "__main__":
    print(json.dumps(intervene(), indent=2))
