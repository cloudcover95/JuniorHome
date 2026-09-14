#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "osai_suite.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.osai_suite import run_all
print(json.dumps(run_all(), indent=2))
