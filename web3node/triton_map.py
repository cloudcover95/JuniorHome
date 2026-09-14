from fuse_kernel import harness
MAP = {"tiling": "n<=256", "autotune": "list vs numpy", "fused_pointwise": "fuse_kernel",
       "fp16_tensorcores": "T1 only"}
def report():
    h = harness(64)
    return {"triton_lang": False, "map": MAP, "fuse_us": h["us"], "t4_ok": h["t4_ok"]}
