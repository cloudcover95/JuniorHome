#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "rails" / "linux" / "asahi" / "vk_sandbox.py").is_file():
        sys.path.insert(0, str(p)); break
from rails.linux.asahi.vk_sandbox import run
print(json.dumps(run(), indent=2))
