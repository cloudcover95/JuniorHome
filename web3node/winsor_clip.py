import numpy as np
def gamma_clip(xs, percentile=95.0):
    a = np.abs(np.asarray(xs, dtype=np.float64).reshape(-1))
    if a.size == 0: return 1.0
    tau = float(np.percentile(a, percentile))
    return float(np.mean(np.minimum(a, tau))) or 1.0
def quant_clip(xs, percentile=95.0):
    g = gamma_clip(xs, percentile)
    out = []
    for x in xs:
        q = round(x / (g + 1e-7))
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return out, g
