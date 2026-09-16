#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "mlkem_note.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.ham_pq import tool
from ports.mlkem_note import status
from ports.zero_trust import metric
note = " ".join(sys.argv[1:]) or "home dash"
print(json.dumps({"kem": status(), "zero_trust": metric(note), "digest": tool(note), "encapsulate": False}, indent=2, default=str))
