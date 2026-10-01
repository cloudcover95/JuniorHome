#!/usr/bin/env python3
"""Scan ~/.juniorhome/deck/inbox. No network. No model pull."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from audio_digest import scan

if __name__ == "__main__":
    print(json.dumps(scan(), indent=2))
