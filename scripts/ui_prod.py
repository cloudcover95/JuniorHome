#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "rails" / "linux" / "home_ui.py").is_file():
        sys.path.insert(0, str(p))
        break
from rails.linux.home_ui import serve

print("http://127.0.0.1:8771")
serve(HOME / "vault").serve_forever()
