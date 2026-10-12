"""Guard the chain id on the trit path. Base rate held, pool held, one pack."""
import sys, time
from juniors_id import hello
from base_rate import BASE
from validate import pack, make, valid

def guard():
    t0 = time.perf_counter()
    row = hello()
    body = pack("home|0.05|1")
    block = valid(make("home|0.05|1"))
    ok = (row["ok"] and row["chain_id"] == "d1fbcc50da6edcc2"
          and row["mint"] is False and row["socket"] is False
          and row["peers"] == 0 and BASE == 0.05
          and block["ok"] and len(body) > 0)
    return {"ok": ok, "chain_id": row["chain_id"], "rate": BASE,
            "pack": len(body), "mint": False, "socket": False,
            "us": round((time.perf_counter() - t0) * 1e6, 1)}

if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
