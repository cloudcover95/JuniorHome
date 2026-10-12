"""Link a path close to the genesis chain id. One trit pack, base rate held."""
import sys, time
from path import path
from juniors_id import hello
from validate import make, valid
from base_rate import BASE

def link():
    t0 = time.perf_counter()
    hid = hello()
    row = path("dev", ["gaia", "home"], True)
    block = make("dev|gaia|home|0.05", hid["chain_id"])
    check = valid(block)
    ok = (hid["ok"] and hid["chain_id"] == "d1fbcc50da6edcc2"
          and row["ok"] and row["rate"] == BASE and row["released"] == 0
          and row["mint"] is False and row["socket"] is False
          and check["ok"] and block["prev"] == hid["chain_id"])
    return {"ok": ok, "chain_id": hid["chain_id"], "link": block["hash"],
            "rate": BASE, "released": 0, "mint": False, "socket": False,
            "us": round((time.perf_counter() - t0) * 1e6, 1)}

if __name__ == "__main__":
    import json
    row = link()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
