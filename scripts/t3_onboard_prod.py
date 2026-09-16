#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "gguf_t3.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.gguf_t3 import run
from ports.fieldcore_spine import expand
drop = Path.home() / ".juniorhome" / "models"
drop.mkdir(parents=True, exist_ok=True)
print(json.dumps({"drop": str(drop), "t3": run("onboard"), "fieldcore": expand(n=32, k=8), "engines_new": False, "download": False}, indent=2, default=str))
