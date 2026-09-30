from junior_gamma import quant_j
from trit5 import pack5
def scan(pressed=None, n=64):
    vec = [0.0] * n
    for i in pressed or []:
        if 0 <= i < n: vec[i] = 1.0
    q, g = quant_j(vec)
    return {"n": n, "down": sorted({i for i in (pressed or []) if 0 <= i < n}),
            "gamma_j": g, "trit5": len(pack5(q)), "mcu": "rp2040-soft", "trit_silicon": False}
