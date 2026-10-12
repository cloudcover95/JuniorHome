"""Guard on the path/record trit-pack. Base rate 0.05 held, mint false, socket false."""
import sys
from record_block import record
from base_rate import BASE

def path_record_guard(actor="forge", hits=None, prev="0" * 16):
    if hits is None:
        hits = ["gaia", "deck"]
    row = record(actor, hits, prev)
    if not row["ok"]:
        return {"ok": False, "why": row.get("why", "record"), "mint": False, "socket": False}
    held = row.get("rate") == BASE and row.get("released", 0) == 0
    good = (
        row["ok"]
        and held
        and row.get("mint") is False
        and row.get("socket", False) is False
        and row.get("in_pack") is True
    )
    return {
        "ok": good,
        "hash": row.get("hash"),
        "prev": row.get("prev"),
        "rate": BASE,
        "held": held,
        "released": 0,
        "juniors": row.get("juniors"),
        "mint": False,
        "socket": False,
        "in_pack": True,
    }

def guard():
    good = path_record_guard()
    bad = path_record_guard("", ["deck"])
    ok = (
        good["ok"]
        and good["rate"] == 0.05
        and good["held"] is True
        and good["mint"] is False
        and good["socket"] is False
        and bad["ok"] is False
    )
    return {
        "ok": ok,
        "hash": good.get("hash"),
        "rate": good.get("rate"),
        "held": good.get("held"),
        "juniors": good.get("juniors"),
        "mint": False,
        "socket": False,
    }

if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
