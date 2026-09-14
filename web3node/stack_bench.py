from bench_winsor import bench as winsor
from fuse_kernel import harness
from trit_ir import build
def main():
    ir = build()
    return {"winsor": winsor(), "fuse64": harness(64), "ir": ir.get("ir"),
            "sandbox_ok": ir.get("sandbox_ok"), "ban": "4096x4096", "trt": False}
