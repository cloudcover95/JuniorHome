"""Junior γ. Does not rewrite bitnet_orig.absmean."""
def gamma_j(xs):
    if not xs: return 1.0
    mag = sorted(abs(x) for x in xs)
    drop = max(1, len(mag)//16)
    core = mag[:-drop] or mag
    return sum(core)/len(core) or 1.0
def quant_j(xs):
    g = gamma_j(xs)
    out = []
    for x in xs:
        q = round(x/g)
        out.append(1 if q>1 else (-1 if q<-1 else int(q)))
    return out, g
