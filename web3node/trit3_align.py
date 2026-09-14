import struct
from bitnet_orig import absmean
from trit5 import pack5, unpack5
HDR = 24
def write(tensor):
    rows = [list(r) for r in tensor]
    flat = [v for r in rows for v in r]
    q, g = absmean(flat)
    hdr = struct.pack("<IIdI", 3, len(rows), float(g), len(rows[0]) if rows else 0) + b"\x00"*4
    assert len(hdr) == HDR
    return hdr + pack5(q)
def read(buf):
    tag, rows, g, cols = struct.unpack_from("<IIdI", buf, 0)
    return {"tag": tag, "shape": (rows, cols), "gamma": g, "n": len(unpack5(buf[HDR:])),
            "payload_off": HDR, "aligned8": HDR % 8 == 0, "bytes": len(buf)}
