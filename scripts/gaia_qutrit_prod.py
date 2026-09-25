#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path(__file__).resolve().parents[1].parent / "JuniorLLM", Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "gaia_qutrit.py").is_file():
        sys.path.insert(0, str(p))
        break
from ports.gaia_qutrit import train
print(json.dumps(train(), indent=2))
