#!/usr/bin/env python3
"""Manager one-shot. Local only. No flash."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from web3node.tuner.manager import run

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "load"
    print(json.dumps(run(mode=mode), indent=2))
