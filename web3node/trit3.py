import struct
from bitnet_orig import absmean
from trit5 import pack5, unpack5
def write(tensor):
    rows = [list(r) for r in tensor]
    flat = [v for r in rows for v in r]
    q, g = absmean(flat)
    return struct.pack("<IIdI", 3, len(rows), float(g), len(rows[0]) if rows else 0) + pack5(q)
def read(buf):
    tag, rows, g, cols = struct.unpack_from("<IIdI", buf, 0)
    q = unpack5(buf[20:])
    return {"tag": tag, "shape": (rows, cols), "gamma": g, "n": len(q)}
