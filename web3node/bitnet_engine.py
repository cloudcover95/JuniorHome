"""BitNet 1.58 engine."""
from __future__ import annotations
import time
from bitnet_orig import absmean, i2s_pack
from schema import append, ticket_doc
from tnn_layer import bitlinear

def infer(x, w, node="T4_ship"):
    t0 = time.perf_counter()
    layer = bitlinear(x, w)
    q, g = absmean(x)
    packed = i2s_pack(q)
    us = (time.perf_counter() - t0) * 1e6
    doc = ticket_doc(layer["y"], g, len(q), node)
    return {"engine": "bitnet-1.58", "y": layer["y"], "gamma": g,
            "pack_bytes": len(packed), "us": round(us, 3), "oauth": False,
            "mongo": False, "jsonl": str(append(doc)), "doc": doc}

if __name__ == "__main__":
    import json
    print(json.dumps(infer([0.2, -0.1, 0.3], [0.4, 0.0, -0.2]), indent=2))
