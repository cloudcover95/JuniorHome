#!/usr/bin/env python3
"""Write Gaia OBJ + blender stub into the Home vault."""
from __future__ import annotations

import json
import sys
from pathlib import Path

for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "gaia_mesh.py").is_file():
        sys.path.insert(0, str(p))
        break

out = Path.home() / ".juniorhome" / "gaia_mesh"
note = " ".join(sys.argv[1:]) or "gaia home portrait"
try:
    from ports.gaia_mesh import write

    print(json.dumps(write(out, note), indent=2))
except Exception as e:
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.txt").write_text("sibling JuniorLLM ports.gaia_mesh missing\n", encoding="utf-8")
    print(json.dumps({"ok": False, "error": type(e).__name__, "dir": str(out), "ue5_launch": False}, indent=2))
