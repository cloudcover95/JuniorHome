"""1.58 opts: reuse Wq, numpy fused when it wins."""
from __future__ import annotations
import math, time
from bitnet_orig import absmean
from tnn_layer import absmax_act, bitlinear
try:
    import numpy as np
    HAVE_NP = True
except Exception:
    HAVE_NP = False
LOG2_3 = math.log2(3)

def fused_reuse(xs, wq, dw):
    xq, _ = absmax_act(xs)
    n = min(len(xq), len(wq))
    acc = sum(a * b for a, b in zip(xq[:n], wq[:n]))
    return acc * (dw / 127.0)

def fused_np(x, w):
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    wa = np.asarray(w, dtype=np.float64).reshape(-1)
    dw = float(np.mean(np.abs(wa))) or 1.0
    wq = np.clip(np.rint(wa / dw), -1, 1)
    am = float(np.max(np.abs(xa))) or 1.0
    xq = np.clip(np.rint(xa * (127.0 / am)), -127, 127)
    n = min(xq.size, wq.size)
    return float(np.dot(xq[:n], wq[:n]) * (dw / 127.0))

def _us(fn, n=80):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6

def bench():
    x = [0.2, -0.1, 0.3, 0.05] * 16
    w = [0.4, 0.0, -0.2, 0.1] * 16
    wq, dw = absmean(w)
    row = {"log2_3": LOG2_3, "store_bits": 2.0, "pad": 2.0 / LOG2_3 - 1.0, "n": len(x),
           "y_ref": bitlinear(x, w)["y"], "y_reuse": fused_reuse(x, wq, dw),
           "list_us": _us(lambda: bitlinear(x, w)), "reuse_us": _us(lambda: fused_reuse(x, wq, dw))}
    if HAVE_NP:
        row["np_us"] = _us(lambda: fused_np(x, w))
        row["agree"] = abs(row["y_ref"] - row["y_reuse"]) < 1e-9
    return row

if __name__ == "__main__":
    import json
    print(json.dumps(bench(), indent=2))
