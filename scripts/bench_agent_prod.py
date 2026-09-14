#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "bench_agent.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.bench_agent import bench
print(json.dumps(bench(), indent=2))
