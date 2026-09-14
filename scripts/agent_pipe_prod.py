#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "agent_pipe.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.agent_pipe import step
print(json.dumps(step(" ".join(sys.argv[1:]) or "buy oats"), indent=2))
