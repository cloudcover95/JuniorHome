#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "gguf_onboard.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.gguf_onboard import onboard
print(json.dumps(onboard(), indent=2, default=str))
