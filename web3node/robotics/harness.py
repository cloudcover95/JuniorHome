"""PC + pose JSON or depth.npy. No vendor SDK."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from trit_quant import absmean, bench, pack

def _depth(n=64):
    yy, xx = np.mgrid[0:n, 0:n]
    return (np.sin(xx / 8.0) + np.cos(yy / 9.0)).astype(np.float64)

def ingest(path=None):
    pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "source": "sim"}
    depth = _depth()
    if path and Path(path).is_file():
        p = Path(path)
        if p.suffix == ".json":
            pose.update(json.loads(p.read_text(encoding="utf-8"))); pose["source"] = str(p)
        elif p.suffix == ".npy":
            depth = np.load(p).astype(np.float64)
    return {"pose": pose, "depth": depth}

def run(path=None):
    frame = ingest(path)
    t0 = time.perf_counter()
    q, g = absmean(frame["depth"])
    packed = pack(q)
    us = (time.perf_counter() - t0) * 1e6
    row = {"pose": frame["pose"], "shape": list(np.asarray(frame["depth"]).shape),
           "absmean_us": round(us, 3), "gamma": g, "ticket_bytes": len(packed),
           "bench": bench(frame["depth"]), "stack_required": False}
    out = Path(__file__).resolve().parent.parent / "vault" / "robot_harness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    row["out"] = str(out)
    return row

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
