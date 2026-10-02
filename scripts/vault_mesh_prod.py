#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from finance_suite import manage
from vault_mesh import write

if __name__ == "__main__":
    manage("web3")
    print(json.dumps(write(), indent=2))
