#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from tritquant import ticket, unpack5, pack5

x = [0.2, -0.4, 0.05, 0.9, -0.1, 0.0, 0.3, -0.7]
w = [0.4, 0.0, -0.2, 0.8, 0.1, -0.05, 0.3, -0.6]
row = ticket(x, w)
wq = unpack5(bytes.fromhex(row["pack5"]), row["n"])
row["roundtrip"] = pack5(wq).hex() == row["pack5"]
print(json.dumps(row, indent=2))
