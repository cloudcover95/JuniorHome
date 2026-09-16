#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "rails" / "linux" / "cuda_align.py").is_file():
        sys.path.insert(0, str(p)); break
from rails.linux.cuda_align import report
print(json.dumps(report(), indent=2))
