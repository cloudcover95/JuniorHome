"""Production-beta latch: T0 root → T4 limb → T1 Spark only for UE."""
from __future__ import annotations
import json
from pathlib import Path
from barebones import tick
from fleet import isolate
from junioros_prod import main as prod

def latch():
    root = isolate({"workload": "m6 home"}, 45)
    limb = isolate({"workload": "robot csi"}, 12)
    spark = isolate({"workload": "ue5 spark"}, 90)
    floor = tick([0.2, -0.1, 0.3], [0.4, 0.0, -0.2], 3.0)
    home = prod()
    row = {"beta": "JuniorOS-latch-0.1", "order": ["T0-root", "T4-limb", "T1-spark-if-ue"],
           "root": {**root, "ticket_on_mini": False}, "limb": {**limb, "ticket_on_mini": False},
           "spark": spark, "floor": {"y": floor["y"], "numpy": floor["numpy"]},
           "prod_ok": home.get("suite_ok"), "ue_on_root": False, "ue_on_limb": False,
           "ue_on_spark": True, "costs_md": "docs/HARDWARE_COSTS.md"}
    out = Path(__file__).resolve().parent / "vault" / "latch.json"
    out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    row["out"] = str(out)
    return row

if __name__ == "__main__":
    print(json.dumps(latch(), indent=2, default=str))
