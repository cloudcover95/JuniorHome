#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "sparsity_ops.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.sparsity_ops import ops
print(json.dumps(ops(" ".join(sys.argv[1:]) or "journal field note"), indent=2, default=str))
