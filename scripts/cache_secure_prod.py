#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "cache_secure.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.cache_secure import put
print(json.dumps(put(" ".join(sys.argv[1:]) or "buy oats"), indent=2))
