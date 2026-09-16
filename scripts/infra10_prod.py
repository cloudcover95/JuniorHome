#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "infra10.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.infra10 import check
print(json.dumps(check(), indent=2))
