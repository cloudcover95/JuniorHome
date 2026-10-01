#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from surface_spin import spin_all

if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "all"
    if kind == "all":
        print(json.dumps(spin_all(), indent=2))
    else:
        from surface_spin import spin
        print(json.dumps(spin(kind), indent=2))
