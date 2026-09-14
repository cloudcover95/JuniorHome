#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "t_layers.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.t_layers import run
layer = "T0"
note = []
for a in sys.argv[1:]:
    if a.startswith("T"): layer = a
    else: note.append(a)
print(json.dumps(run(layer, " ".join(note) or "home dash"), indent=2, default=str))
