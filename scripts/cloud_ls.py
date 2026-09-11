#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "scripts" / "cloud_ls.py").is_file():
        sys.path.insert(0, str(p))
        break
from scripts.cloud_ls import main

raise SystemExit(main(["cloud_ls", str(HOME / "vault")]))
