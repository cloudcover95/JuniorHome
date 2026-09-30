#!/usr/bin/env python3
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from deck_port import log_note
print(json.dumps(log_note("cad mount"), indent=2))
