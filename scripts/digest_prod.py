#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from digest import digest

if __name__ == "__main__":
    note = sys.argv[1] if len(sys.argv) > 1 else "JuniorHome"
    print(json.dumps(digest(note), indent=2))
