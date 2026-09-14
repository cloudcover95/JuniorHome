import os
from fuse_kernel import fused
from home_kernel import probe
HOOKS = {
    "cpu": "JUNIOR_ORT_CPU", "cuda": "JUNIOR_ORT_CUDA", "trt": "JUNIOR_ORT_TRT",
    "coreml": "JUNIOR_ORT_COREML", "nnapi": "JUNIOR_ORT_NNAPI", "azure": "JUNIOR_ORT_AZURE",
}
def hooks():
    return {"default": "fused-list-1.58",
            "armed": {k: bool(os.environ.get(v)) for k, v in HOOKS.items()},
            "cuda_cli": bool(probe().get("cuda_cli")),
            "y": fused([0.2,-0.1,0.3],[0.4,0.0,-0.2])["y"],
            "compat": list(HOOKS)}
