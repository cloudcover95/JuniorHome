import time
from fuse_kernel import fused
def _us(fn, k=40):
    t0 = time.perf_counter()
    for _ in range(k): fn()
    return (time.perf_counter()-t0)/k*1e6
def bench(n):
    x = [((i*3)%7-3)/4.0 for i in range(n)]
    w = [((i*5)%9-4)/5.0 for i in range(n)]
    return {"n": n, "fused_us": _us(lambda: fused(x, w)), "ort": False, "ort_wins": False}
