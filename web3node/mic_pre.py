from junior_gamma import gamma_j
def pre(samples, gain_db=0.0):
    g = gamma_j(samples)
    lin = 10 ** (gain_db / 20.0)
    out = [max(-1.0, min(1.0, x * lin)) for x in samples]
    peak = max((abs(x) for x in out), default=0.0)
    return {"gamma_j": g, "gain_db": gain_db, "peak": peak, "n": len(out), "clip": peak>=1.0, "hw_pre": False}
