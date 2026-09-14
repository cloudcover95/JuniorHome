from trit5 import pack5
from trit_on import schema
from tnn_layer import bitlinear
def room(name, x, w):
    layer = bitlinear(x, w)
    return {"room": name, "y": layer["y"], "n": layer["n"]}
def palace(ask="field"):
    x, w = [0.2, -0.1, 0.3], [0.4, 0.0, -0.2]
    return {"sis": True, "palace": True, "math": schema()["name"],
            "rooms": [room("infer", x, w), room("ticket", x, w)], "ask": ask}
