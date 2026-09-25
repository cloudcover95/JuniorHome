#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "ros2_dds.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.ros2_dds import status
print(json.dumps(status(), indent=2))
