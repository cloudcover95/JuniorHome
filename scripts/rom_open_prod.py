#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from rom_open import open_rom

if __name__ == "__main__":
    do = "--open" in sys.argv
    row = open_rom(do)
    print(json.dumps(row, indent=2))
    sys.exit(0 if row.get("ok") or row.get("why") in {"no_rom", "no_core"} else 1)
