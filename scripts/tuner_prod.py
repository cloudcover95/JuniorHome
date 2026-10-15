#!/usr/bin/env python3
"""One-shot: pack a sample, write mesh line. No flash. No model."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from web3node.tuner.channels import pack_sample
from web3node.tuner.dyno import power_from_force, pack_run

def main():
    sample = {"rpm": 4500, "afr": 14.7, "tps": 0.4, "map": 1.0, "ect": 90, "iat": 30, "vss": 0, "gear": 3, "analog": 0.1}
    ch = pack_sample(sample)
    dy = power_from_force(4500, 200.0, 0.25)
    run = pack_run(dy)
    mesh = Path.home() / ".juniorhome" / "gaia_mesh"
    mesh.mkdir(parents=True, exist_ok=True)
    line = {"kind": "tuner", "channels": ch, "dyno": run}
    (mesh / "tuner.jsonl").open("a", encoding="utf-8").write(json.dumps(line) + "\n")
    print(json.dumps({"ok": True, "close": ch["close"], "power_hp": dy["power_hp"]}, indent=2))

if __name__ == "__main__":
    main()
