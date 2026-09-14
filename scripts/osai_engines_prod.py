#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "osai_engines.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.osai_engines import all_three, run
args = sys.argv[1:]
eng = args[0] if args and args[0] in {"vault","media","notes"} else None
note = " ".join(args[1:] if eng else args) or "journal field note"
print(json.dumps(run(eng, note) if eng else all_three(note), indent=2))
