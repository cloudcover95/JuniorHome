#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path(__file__).resolve().parents[1].parent / "JuniorLLM", Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "gell_mann.py").is_file():
        sys.path.insert(0, str(p))
        break
from ports.gell_mann import proofs
from ports.pulse_lake import append, tail
from ports.qutrit_qec import decode, encode
from ports.qutrit_rotate import demo

append(1, 1.5708, 1.0)
print(json.dumps({"proofs": proofs(), "qec": decode(encode(1)), "ry": demo(), "lake": tail(2)}, indent=2))
