#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "zt_net.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.zt_net import net
print(json.dumps(net(" ".join(sys.argv[1:]) or "home dash"), indent=2))
