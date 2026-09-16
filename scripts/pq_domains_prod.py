#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "pq_domains.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.pq_domains import domains
print(json.dumps(domains(), indent=2))
