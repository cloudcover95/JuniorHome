#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path(__file__).resolve().parents[1].parent / "JuniorLLM", Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "operator_box.py").is_file():
        sys.path.insert(0, str(p))
        break
from ports.operator_box import collab, emu_tick
print(json.dumps({"collab": collab(["home dash ok", "gaia spine ok", "fail bind"]), "emu": emu_tick(1)}, indent=2))
