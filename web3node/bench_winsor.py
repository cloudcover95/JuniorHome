import time, numpy as np
from bitnet_orig import absmean
from junior_gamma import quant_j
from trit5 import pack5
from winsor_clip import quant_clip
def _us(fn, n=40):
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter()-t0)/n*1e6
def row(n, spike=True):
    x = np.random.default_rng(7).normal(size=n).tolist()
    if spike: x[n//2] = 8.0
    qb, gb = absmean(x); qj, gj = quant_j(x); qc, gc = quant_clip(x, 95)
    return {"n": n, "g_abs": gb, "g_drop": gj, "g_clip": gc,
            "nz_abs": sum(1 for v in qb if v), "nz_drop": sum(1 for v in qj if v),
            "nz_clip": sum(1 for v in qc if v),
            "us_abs": _us(lambda: absmean(x)), "us_drop": _us(lambda: quant_j(x)),
            "us_clip": _us(lambda: quant_clip(x)), "trit5_B": len(pack5(qj))}
def bench():
    return {"T4_64": row(64), "T4_256": row(256), "T0_1024": row(1024), "not": "4096x4096"}
