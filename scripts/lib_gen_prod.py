#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "lib_gen.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.lib_gen import generate
print(json.dumps(generate(sys.argv[1] if len(sys.argv) > 1 else "mjbatch", " ".join(sys.argv[2:]) or "host"), indent=2))
