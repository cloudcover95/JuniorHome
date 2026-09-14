"""GGUF-v3 wrapper around trit I2_S payload. Not Microsoft 2B."""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np
from bitnet_orig import absmean
from trit_cache import pack_trits
GGUF, VER, GTYPE_U8 = b"GGUF", 3, 8

def _s(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b

def write_i2s(weights, path=None):
    q, g = absmean(weights)
    blob = pack_trits(np.clip(np.rint(np.asarray(q)), -1, 1))
    path = path or Path(__file__).resolve().parent / "vault" / "juniorosai.i2s.gguf"
    body = bytearray()
    body += GGUF + struct.pack("<IQQ", VER, 1, 3)
    body += _s("general.architecture") + struct.pack("<I", 8) + _s("juniorosai")
    body += _s("general.quantization_version") + struct.pack("<I", 8) + _s("i2s-trit-jtr1")
    body += _s("juniorosai.gamma") + struct.pack("<I", 12) + struct.pack("<d", float(g))
    body += _s("blk.0.weight") + struct.pack("<I", 1) + struct.pack("<Q", len(blob))
    body += struct.pack("<I", GTYPE_U8) + struct.pack("<Q", 0)
    body += b"\x00" * ((32 - (len(body) % 32)) % 32) + blob
    path.write_bytes(bytes(body))
    return {"path": str(path), "bytes": path.stat().st_size, "n": len(q), "gamma": g,
            "microsoft_2b": False, "llama_cpp_loadable": False}

if __name__ == "__main__":
    import json
    print(json.dumps(write_i2s([0.2, -0.4, 0.1, 0.0] * 8), indent=2))
