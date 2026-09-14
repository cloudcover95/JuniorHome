import struct
import numpy as np
QK = 256
def pack(x):
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    out = bytearray(struct.pack("<I", a.size))
    for i in range(0, a.size, QK):
        blk = a[i:i+QK]
        if blk.size < QK:
            blk = np.pad(blk, (0, QK-blk.size))
        scale = float(np.max(np.abs(blk))/7.0) or 1.0
        q = np.clip(np.rint(blk/scale), 0, 15).astype(np.uint8)
        out += struct.pack("<f", scale)
        for j in range(0, QK, 2):
            out.append(int(q[j] | (q[j+1]<<4)))
    return bytes(out)
def report(n=256):
    blob = pack(np.random.default_rng(4).normal(size=n).astype(np.float32))
    return {"n": n, "q4k_shaped_bytes": len(blob), "float32": n*4, "qk": QK, "ggml_exact": False}
