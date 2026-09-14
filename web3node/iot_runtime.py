from fuse_kernel import fused
from home_kernel import probe
def pick():
    host = probe()
    x, w = [0.2, -0.1, 0.3], [0.4, 0.0, -0.2]
    return {"runtime": "fused-list", "iot": True, "cuda_cli": bool(host.get("cuda_cli")),
            "mlx": bool(host.get("mlx")), "y": fused(x, w)["y"],
            "t0_onnx": "bitlinear_onnx.py on T0 only"}
