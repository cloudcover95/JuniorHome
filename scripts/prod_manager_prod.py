#!/usr/bin/env python3
"""Production manager one-shot. Local only."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from web3node.prod_manager import run

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
