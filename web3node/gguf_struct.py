"""Walk GGUF v3 we emit + JTR1."""
from __future__ import annotations
import struct
from pathlib import Path
from gguf_i2s import write_i2s
from tritpack import write_pack
GGUF_VAL = {8: "string", 12: "float64"}

def _str(buf, i):
    n, = struct.unpack_from("<Q", buf, i); i += 8
    return buf[i:i+n].decode("utf-8", "replace"), i+n

def parse_gguf(path):
    raw = Path(path).read_bytes()
    ver, n_ten, n_kv = struct.unpack_from("<IQQ", raw, 4)
    i = 24; kvs = []
    for _ in range(n_kv):
        key, i = _str(raw, i)
        vt, = struct.unpack_from("<I", raw, i); i += 4
        if vt == 8: val, i = _str(raw, i)
        elif vt == 12:
            val, = struct.unpack_from("<d", raw, i); i += 8
        else: val = vt
        kvs.append({"k": key, "t": GGUF_VAL.get(vt, vt), "v": val})
    tens = []
    for _ in range(n_ten):
        name, i = _str(raw, i)
        nd, = struct.unpack_from("<I", raw, i); i += 4
        dims = list(struct.unpack_from("<"+"Q"*nd, raw, i)); i += 8*nd
        gtype, off = struct.unpack_from("<IQ", raw, i); i += 12
        tens.append({"name": name, "dims": dims, "gtype": gtype, "offset": off})
    pad = (32 - (i % 32)) % 32
    return {"magic": raw[:4].decode(), "ver": ver, "kv": kvs, "tensors": tens,
            "header_end": i, "pad": pad, "payload_bytes": len(raw)-i-pad, "file_bytes": len(raw)}

def parse_jtr1(path):
    raw = Path(path).read_bytes()
    ver, g, n, nb = struct.unpack_from("<IdII", raw, 4)
    return {"magic": raw[:4].decode(), "ver": ver, "gamma": g, "n": n,
            "payload_bytes": nb, "file_bytes": len(raw), "header_bytes": 24}

def report():
    w = [0.2, -0.4, 0.1, 0.0]*8
    return {"gguf": parse_gguf(write_i2s(w)["path"]), "jtr1": parse_jtr1(write_pack(w)["path"])}

if __name__ == "__main__":
    import json
    print(json.dumps(report(), indent=2))
