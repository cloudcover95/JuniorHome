#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "operator_box.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.operator_box import status
from ports.persist_view import view
print(json.dumps({"box": status(), "persist": view()}, indent=2))
