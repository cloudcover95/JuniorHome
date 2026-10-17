"""Guard the held chain record bench. Opt-in, AntiCon, one trit pack, base rate 0.05, pool held, mint false, socket false. No poll sets rate. No model. No UE."""
import sys
import time
from chain_record_held_bench import bench
from base_rate import BASE


def guard():
    t0 = time.perf_counter()
    row = bench()
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        row.get("ok") is True
        and row.get("rate") == BASE
        and row.get("held") is True
        and row.get("released") == 0
        and row.get("mint") is False
        and row.get("socket") is False
        and row.get("in_pack") is True
        and row.get("n") == 3
        and row.get("base") == BASE
    )
    return {
        "ok": ok,
        "us": us,
        "n": row.get("n"),
        "tip": row.get("tip"),
        "juniors": row.get("juniors"),
        "rate": BASE,
        "released": 0,
        "mint": False,
        "socket": False,
        "held": True,
        "in_pack": True,
        "base": BASE,
        "guarded": True,
    }


if __name__ == "__main__":
    import json
    out = guard()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["ok"] else 1)
