"""JuniorHome sandbox suite — numpy clock."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Callable
import numpy as np
from compute_profiles import pick
from home_kernel import backend, dispatch, probe
from junioros_rails import architecture
from off_caps import all_off, refetch
from tnn_layer import bitlinear
from trit_cache import compare

def _us(fn: Callable, n: int = 40) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6

def _case(name: str, fn: Callable) -> dict[str, Any]:
    try:
        return {"name": name, "ok": True, "detail": fn()}
    except Exception as exc:
        return {"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

def run() -> dict[str, Any]:
    host = probe()
    rng = np.random.default_rng(1)
    x = rng.normal(size=64).tolist()
    w = rng.normal(size=64).tolist()
    mat = np.clip(np.rint(rng.normal(size=(16, 16))), -1, 1)
    cases = [
        _case("probe_numpy_clock", lambda: {"backend": backend(host), "mlx": host.get("mlx"),
              "expect_numpy": backend(host) == "numpy" and not host.get("mlx")}),
        _case("kernel_t0_no_ue", lambda: dispatch("sandbox numpy", "agent", 45)["ue_boot"] is False),
        _case("profile_ue_stays_t0_at_45w", lambda: pick("ue5 spark", 45) == "T0_home"),
        _case("bitlinear", lambda: bitlinear(x, w)["n"] == 64),
        _case("trit_pack_smaller", lambda: compare(mat)["trit_pack_bytes"] < compare(mat)["float32_bytes"]),
        _case("off_rails_all_off", lambda: all(v.get("state") in {"off", "allowed"} for v in all_off().values())),
        _case("refetch_blocked", lambda: refetch("epqs")["state"] == "off" and refetch("yahoo")["state"] == "off"),
        _case("rails_contract", lambda: architecture()["fake_gpu"] is False),
    ]
    benches = {
        "bitlinear_64_us": round(_us(lambda: bitlinear(x, w), 50), 3),
        "trit_pack_16_us": round(_us(lambda: compare(mat), 30), 3),
        "dispatch_us": round(_us(lambda: dispatch("sandbox", "agent", 45), 20), 3),
        "probe_us": round(_us(probe, 20), 3),
    }
    failed = [c["name"] for c in cases if not c["ok"]]
    row = {"suite": "juniorhome-sandbox-numpy", "host": host, "clock": backend(host),
           "benches_us": benches, "cases": cases, "passed": len(cases) - len(failed),
           "failed": failed, "ok": not failed}
    out = Path(__file__).resolve().parent / "vault" / "sandbox_suite.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    row["out"] = str(out)
    return row

if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, default=str))
    raise SystemExit(0 if result["ok"] else 1)
