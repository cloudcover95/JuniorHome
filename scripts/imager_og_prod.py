#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "imager_og.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.imager_og import image_og
print(json.dumps(image_og(" ".join(sys.argv[1:]) or "home vault memory"), indent=2, default=str))
