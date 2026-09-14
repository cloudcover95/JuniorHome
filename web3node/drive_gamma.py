from bitnet_orig import absmean
from junior_gamma import quant_j
from trit3_align import write as trit3_write
def residual(depth):
    return [depth[i]-depth[i-1] for i in range(1, len(depth))] if len(depth)>1 else depth
def fuse(depth):
    r = residual(depth)
    qb, gb = absmean(r)
    qj, gj = quant_j(r)
    return {"bitnet_g": gb, "junior_g": gj, "q_junior": qj[:8],
            "trit3_bytes": len(trit3_write([r[:3] or [0,0,0], r[3:6] or [0,0,0]])),
            "fsd": False, "gamma_rays": False, "hid": "crispy-mouse PIO"}
