"""Memory bench. Open envelope stays in RAM. No 1024 SVD. No order."""
import time
from pathlib import Path
from write_gate import commit
def absmean(xs):
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / gamma)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return out, gamma
def tickets():
    w = [0.2, -0.1, 0.4, 0.0, -0.3, 0.5, 0.1, -0.2]
    q, g = absmean(w)
    return {"home": "engine hold", "web3node": f"gamma={g:.4f} zeros={q.count(0)}",
            "juniorstock": "return ticket no order", "osai_gaia": "goldend-osai-omega/1 hop"}
def bench(env="open"):
    t0 = time.perf_counter()
    rows = []
    for name, text in tickets().items():
        path = Path.home() / ".juniorhome" / "ram" / f"{name}.txt"
        row = commit(env, path, text)
        rows.append({"port": name, "disk": row["disk"], "why": row["why"], "n": len(text)})
    return {"env": env, "us": round((time.perf_counter() - t0) * 1e6, 1), "rows": rows,
            "svd_1024": False, "order": False, "model_pull": False}
