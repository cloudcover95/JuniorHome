"""TNN twin of junior_bitnet.bitlinear. Live file stays source of truth."""
from __future__ import annotations
from bitnet_orig import absmean

def absmax_act(xs: list[float]) -> tuple[list[int], float]:
    am = max((abs(x) for x in xs), default=1.0) or 1.0
    scale = 127.0 / am
    out = []
    for x in xs:
        q = int(round(x * scale))
        out.append(-127 if q < -127 else (127 if q > 127 else q))
    return out, scale

def bitlinear(x: list[float], w: list[float]) -> dict:
    wq, dw = absmean(w)
    xq, dx = absmax_act(x)
    n = min(len(xq), len(wq))
    acc = sum(a * b for a, b in zip(xq[:n], wq[:n]))
    return {"y": acc * (dw / 127.0), "acc": acc, "dw": dw, "dx": dx, "n": n}
