#!/usr/bin/env python3
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.lab_floor import spin
print(json.dumps(spin("--off" not in sys.argv), indent=2))
