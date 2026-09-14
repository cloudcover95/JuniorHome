from fuse_kernel import fused
from home_kernel import backend, probe
from ship import card
def generate(prompt):
    host = probe()
    return {"status": "0x00", "compute_node": backend(host), "mlx": bool(host.get("mlx")),
            "prompt": prompt[:80], "y": fused([0.2,-0.1,0.3],[0.4,0.0,-0.2])["y"],
            "ship": card()["ship"]}
