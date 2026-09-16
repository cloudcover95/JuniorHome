#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "osai_suite.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.osai_suite import run_all
r = run_all()
print(json.dumps({k: r[k] for k in ("suite", "passed", "n", "download") if k in r}, indent=2))
print(json.dumps({"suites": [{s["file"]: f"{s['passed']}/{s['n']}"} for s in r.get("suites") or []]}, indent=2))
sys.exit(0 if r.get("passed") == r.get("n") else 1)
