"""5 trits/byte. 3**5=243."""
from bitnet_orig import absmean
from trit_cache import pack_trits
import numpy as np

def pack5(trits):
    t = [0 if x == 0 else (1 if x > 0 else 2) for x in trits]
    out = bytearray()
    for i in range(0, len(t), 5):
        acc, p = 0, 1
        for d in t[i:i+5]:
            acc += d * p; p *= 3
        out.append(acc)
    return bytes([len(trits)&255, (len(trits)>>8)&255]) + bytes(out)

def unpack5(blob):
    n = blob[0] | (blob[1]<<8); vals = []
    for b in blob[2:]:
        for _ in range(5):
            d = b % 3; b //= 3
            vals.append(0 if d == 0 else (1 if d == 1 else -1))
    return vals[:n]

def report(n=64):
    xs = [((i*17)%7-3)/3.0 for i in range(n)]
    q, g = absmean(xs)
    p2 = pack_trits(np.clip(np.rint(q), -1, 1))
    p5 = pack5(q)
    return {"n": n, "gamma": g, "bit2": len(p2), "trit5": len(p5),
            "float32": n*4, "roundtrip": unpack5(p5)==q, "bits_per_trit5": 8/5}

if __name__ == "__main__":
    import json
    print(json.dumps(report(), indent=2))
