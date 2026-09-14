from fuse_kernel import fused
NODES = {"T4": {"pair": False}, "T0": {"pair": True}, "linux": {"pair": True}, "T1": {"pair": True}}
def route(kind):
    k = (kind or "").lower()
    if "ue" in k or "120b" in k or "trt" in k: return "T1"
    if "ticket" in k or "iot" in k: return "T4"
    return "T0"
def domain(ask="agent tick"):
    node = route(ask)
    return {"node": node, "pair_proxy": NODES[node]["pair"], "vram_pool": False, "split_120b": False}
