#!/usr/bin/env python3
import json, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for p in (root.parent / "JuniorLLM", Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "trit_thread.py").is_file():
        sys.path.insert(0, str(p))
        break
from ports.trit_thread import ham_h0, thread

xs = [float(i) for i in range(32)]
ys = list(reversed(xs))
a, b = thread(xs), thread(ys)
print(json.dumps({"up": a, "down": b, "h0_r2": ham_h0([a["i2s_cen"], b["i2s_cen"]], 2)}, indent=2))
