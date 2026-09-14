#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "gguf_t3.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.gguf_t3 import run
print(json.dumps(run(" ".join(sys.argv[1:]) or "t3"), indent=2))
