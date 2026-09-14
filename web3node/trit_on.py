"""Trit-on math. Not NVIDIA Triton."""
import math
LOG2_3 = math.log2(3)
def schema():
    return {"name": "trit-on-1.58", "triton_gpu": False, "log2_3": LOG2_3,
            "store_bits": 2.0, "pad": 2.0/LOG2_3-1.0, "y": "(Xq·Wq)*(γ/127)"}
