import json
from pathlib import Path
def export(n=64):
    graph = {"ir_version": 8, "opset": None, "n": n, "float16_4096": False,
             "nodes": [{"op": "AbsMean"}, {"op": "RoundClip"}, {"op": "AbsMax"},
                        {"op": "Dot"}, {"op": "Scale"}]}
    path = Path(__file__).resolve().parent / "vault" / "bitlinear.onnx.json"
    path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return {"path": str(path), "onnx_pkg": False, "n": n}
