import json
from pathlib import Path
from fuse_kernel import fused
from home_terraform import inject
def capture(x, w):
    return {"graph": ["absmean_w", "absmax_x", "ternary_w", "dot", "scale_y"],
            "cuda": False, "trt": False, "replay": fused(x, w)}
def plan():
    cap = capture([0.2, -0.1, 0.3], [0.4, 0.0, -0.2])
    spec = {"op": "BitLinear-1.58", "plugin": "JuniorOSaiTritOn", "onnx_opset": None,
            "cuda_graph": cap["graph"], "commercial": ["TensorRT", "TensorRT-LLM"],
            "where": "T1_spark", "built": False}
    path = Path(__file__).resolve().parent / "vault" / "trt_plan.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return {**spec, "path": str(path), "replay_y": cap["replay"]["y"],
            "port": inject("juniorosai tensorrt spark").get("port")}
