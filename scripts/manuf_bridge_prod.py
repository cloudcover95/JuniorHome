#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "manuf_bridge.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.manuf_bridge import bridge
print(json.dumps(bridge(" ".join(sys.argv[1:]) or "dxf tendon cad"), indent=2, default=str))
