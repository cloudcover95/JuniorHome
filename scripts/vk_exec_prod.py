#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "rails" / "linux" / "asahi" / "vk_exec.py").is_file():
        sys.path.insert(0, str(p)); break
from rails.linux.asahi.vk_exec import exec_shader
print(json.dumps(exec_shader(), indent=2))
