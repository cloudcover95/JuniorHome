#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "agent_flows.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.agent_flows import all_flows, run
args = sys.argv[1:]
name = args[0] if args and args[0] in {"dash","spine","cad","vault","pipe"} else None
note = " ".join(args[1:] if name else args) or "home dash"
print(json.dumps(run(name, note) if name else all_flows(note), indent=2, default=str))
