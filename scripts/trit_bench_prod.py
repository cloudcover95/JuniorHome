#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "junior_bitnet" / "trit_bench.py").is_file():
        sys.path.insert(0, str(p)); break
from junior_bitnet.trit_bench import run
print(json.dumps(run(), indent=2))
