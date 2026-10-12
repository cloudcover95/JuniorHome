"""One ticket for the stack that exists."""
import sys, time
from chain_run_bench import bench as chain_bench
from path_bench import bench as path_bench
from prove_chain import prove
from record_chain import chain
def status():
    t0 = time.perf_counter()
    rows = {"path": path_bench()["ok"], "chain": chain_bench()["ok"],
            "prove": prove([("dev", ["gaia"]), ("home", ["prompt"])])["ok"],
            "record": chain()["ok"]}
    return {"ok": all(rows.values()), "rows": rows,
            "us": round((time.perf_counter() - t0) * 1e6, 1),
            "mint": False, "socket": False}
if __name__ == "__main__":
    import json
    row = status()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
