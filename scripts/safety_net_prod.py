#!/usr/bin/env python3
"""Safety net one-shot. Local only."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from web3node.safety_net import check

if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
