#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "osai_suite.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.gaia_proto import handshake
from ports.gguf_t3 import run as t3
from ports.osai_suite import run_all
from ports.fieldcore_spine import expand
suite = run_all()
hs = handshake("home dash", job="dash-viewport")
print(json.dumps({
    "osai": {"passed": suite.get("passed"), "n": suite.get("n")},
    "agi": {"protocol": hs.get("protocol"), "schema_ok": hs.get("schema_ok")},
    "t3": t3("onboard"),
    "fieldcore": {"n": expand(n=32, k=8).get("n")},
    "sdk": "AGI_SDK pointer",
    "rust_ffi": False,
    "download": False,
}, indent=2, default=str))
sys.exit(0 if suite.get("passed") == suite.get("n") and hs.get("schema_ok") else 1)
