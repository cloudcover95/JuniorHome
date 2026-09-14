from pathlib import Path
import time
from fieldcore_bridge import pick_juniorllm_port
from second_brain import write_note
from trit5 import pack5
from tnn_layer import absmax_act
def fused(x, w):
    n = min(len(x), len(w))
    g = (sum(abs(w[i]) for i in range(n))/n) if n else 1.0
    xq, _ = absmax_act(x[:n])
    acc, q = 0, []
    for i in range(n):
        t = round(w[i]/g)
        t = 1 if t>1 else (-1 if t<-1 else int(t))
        q.append(t); acc += xq[i]*t
    return {"y": acc*(g/127.0), "gamma": g, "n": n, "trit5": len(pack5(q)), "path": "fused-list"}
def harness(n=64):
    x = [((i*3)%7-3)/4.0 for i in range(n)]
    w = [((i*5)%9-4)/5.0 for i in range(n)]
    t0 = time.perf_counter()
    for _ in range(50): fused(x, w)
    us = (time.perf_counter()-t0)/50*1e6
    row = fused(x, w)
    return {"row": row, "us": us, "trt": False, "t4_ok": n<=256,
            "port": pick_juniorllm_port("juniorosai fuse"),
            "brain": str(write_note(Path(__file__).resolve().parent/"vault", f"# fuse {n}\n"))}
