from pathlib import Path
import numpy as np, onnx
from onnx import TensorProto, helper
import onnxruntime as ort
from fuse_kernel import fused

def build(n=64):
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [n])
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [n])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [])
    inits = [
        helper.make_tensor("c127", TensorProto.FLOAT, [], [127.0]),
        helper.make_tensor("wlo", TensorProto.FLOAT, [], [-1.0]),
        helper.make_tensor("whi", TensorProto.FLOAT, [], [1.0]),
        helper.make_tensor("xlo", TensorProto.FLOAT, [], [-127.0]),
        helper.make_tensor("xhi", TensorProto.FLOAT, [], [127.0]),
    ]
    nodes = [
        helper.make_node("Abs", ["W"], ["Wabs"]),
        helper.make_node("ReduceMean", ["Wabs"], ["gamma"], keepdims=0),
        helper.make_node("Div", ["W", "gamma"], ["Wdiv"]),
        helper.make_node("Round", ["Wdiv"], ["Wround"]),
        helper.make_node("Clip", ["Wround", "wlo", "whi"], ["Wq"]),
        helper.make_node("Abs", ["X"], ["Xabs"]),
        helper.make_node("ReduceMax", ["Xabs"], ["xmax"], keepdims=0),
        helper.make_node("Div", ["c127", "xmax"], ["xscale"]),
        helper.make_node("Mul", ["X", "xscale"], ["Xscaled"]),
        helper.make_node("Round", ["Xscaled"], ["Xround"]),
        helper.make_node("Clip", ["Xround", "xlo", "xhi"], ["Xq"]),
        helper.make_node("Mul", ["Xq", "Wq"], ["prod"]),
        helper.make_node("ReduceSum", ["prod"], ["acc"], keepdims=0),
        helper.make_node("Div", ["gamma", "c127"], ["gscale"]),
        helper.make_node("Mul", ["acc", "gscale"], ["Y"]),
    ]
    model = helper.make_model(helper.make_graph(nodes, "JuniorOSaiTritOn", [W, X], [Y], inits),
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model

def run(n=64):
    path = Path(__file__).resolve().parent / "vault" / "bitlinear.onnx"
    path.write_bytes(build(n).SerializeToString())
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.array([((i*3)%7-3)/4.0 for i in range(n)], dtype=np.float32)
    w = np.array([((i*5)%9-4)/5.0 for i in range(n)], dtype=np.float32)
    y = float(sess.run(["Y"], {"X": x, "W": w})[0])
    ref = fused(x.tolist(), w.tolist())["y"]
    return {"bytes": path.stat().st_size, "y_ort": y, "y_fused": ref,
            "agree": abs(y-ref)<1e-4, "trt": False, "cuda": False, "n": n}
